import argparse
import csv
import datetime
import glob
import importlib
import os
import sys
import time
from functools import partial
from typing import Any

import neptune as neptune
import boto3
import omegaconf

from aiosynawsmodules.services.s3 import upload_file
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from get_data import download_dataset
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config, plot_images, sync_logdir, CustomModelCheckpoint, attempt_key_read, \
    count_params
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Neptune Logging
from pytorch_lightning.loggers import NeptuneLogger

# from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from aiosynawsmodules.services.s3 import download_directory, download_file, upload_directory
from aiosynawsmodules.services.sso import set_sso_profile

# from pytorch_lightning.plugins import DDPPlugin

os.environ["WANDB_SILENT"] = "true"


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=24,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--neptune_mode", type=str, default="async", help="mode neptune should run in (sync, async or debug or ...)"
    )

    parser.add_argument("--location", type=str, default="local", help="Run locally (laptop) or remote (aws)")
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser) #Only works with lightning 1.4
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size: (worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            location,
            train=None,
            validation=None,
            test=None,
            predict=None,
            wrap=False,
            num_workers=None,
            shuffle_test_loader=False,
            use_worker_init_fn=False,
            shuffle_val_dataloader=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.location = location
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # Add the location to each of the configs.
        for entry in self.dataset_configs:
            self.dataset_configs[entry]["params"]["config"]["location"] = self.location

        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=init_fn,
        )

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=False,
        )

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets["predict"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["predict"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
        )


class ThesisCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any,
            batch_idx: int
    ) -> None:
        trainer.logger.log_metrics({"train/Loss": outputs["loss"]}, step=trainer.global_step)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("THESISCALLBACK: Training is starting...")
        # Log the number of trainable params
        params = count_params(trainer.model.model._modules['diffusion_model'])
        trainer.logger.experiment["Trainable Parameters (unet)"] = f"{params:_}"

    def on_train_end(self, trainer, pl_module):
        print("Training completed.")

    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        lr = trainer.model.optimizers().param_groups[0]["lr"]
        trainer.logger.log_metrics({"lr-abs": lr}, step=trainer.global_step)
        # log the batch every 5000 steps ?
        if trainer.global_step % 5000 == 0:
            plot_images(trainer, batch["image"], batch_idx, 4, len(batch["image"]) // 4)
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Potentially fun to sample a single image after every, say, 10 epochs with the same caption to see it
        # hopefully progress

        print(f"Finished the epoch, global step:{trainer.global_step}.")
        ckpt_path = os.path.join(ckptdir, f"end_epoch_{trainer.current_epoch}")
        trainer.save_checkpoint(ckpt_path, weights_only=False)
        print(f"Saved checkpoint at {ckpt_path}")
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # print(f"Starting epoch {trainer.current_epoch}. ")
        lr = trainer.model.optimizers().param_groups[0]["lr"]
        print(f"{lr = }, {trainer.current_epoch = }")

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Starting validation ...")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        try:
            samples = trainer.model.validation_step_outputs[0]  # A batch of 8/16 imgs, it has shape (8,3,size), dtype uint8)
            samples_t = torch.from_numpy(samples / 255.0)
            val_inputs = trainer.model.validation_step_inputs[0]
        except IndexError:
            # Sometimes happens when resuming a run in an unfortunate position
            samples_t = torch.rand((16,3,trainer.model.image_size * 4, trainer.model.image_size * 4))
            val_inputs = "Not a generated sample. Randomly sampled noise due to resuming a run in an unfortunate position."


        grid = torchvision.utils.make_grid(samples_t, nrow=4, padding=10)
        grid = transforms.functional.to_pil_image(grid)

        trainer.logger.experiment[f"validation_samples/generated_images"].log(
            grid,
            name=f"Validation sample created at epoch {trainer.current_epoch} and step {trainer.global_step}.",
            description=f'Size of image: {list(samples_t[0].shape)}. Example of caption used: {val_inputs}.',
        )
        # Sync the whole logdirectory with aws, so upload it and overwrite is ok
        print("Syncing logdir from val epoch end...")
        sync_logdir(opt, trainer, logdir)


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%m-%dT%H-%M")
    torch.set_float32_matmul_precision("highest") # Changed this

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    sys.path.append(os.getcwd())
    if opt.location in ["maclocal"]:
        taming_dir = os.path.abspath("src/taming-transformers")
    elif opt.location in ["local"]:
        taming_dir = os.path.abspath("src/taming-transformers")
    elif opt.location == "remote":
        taming_dir = os.path.abspath("code/generationLDM/src/taming-transformers")
    else:
        assert False, "Unknown location"
    sys.path.append(taming_dir)

    # Set Neptune mode, project and API key
    os.environ[
        "NEPTUNE_API_TOKEN"
    ] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYTRjNmEyNy1lNTY5LTRmYTMtYjg5Yy03YjIxOTNhN2MwNGQifQ\=\="
    os.environ["NEPTUNE_PROJECT"] = "aiosyn/generation"
    os.environ["NEPTUNE_MODE"] = opt.neptune_mode

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:

        if opt.resume[-1] == "/":
            # Remove final "/"
            opt.resume = opt.resume[:-1]

        logdir = "logs"
        if opt.location in ['local', 'maclocal']:
            set_sso_profile(profile_name="aws-aiosyn-data", region_name="eu-west-1")

        # Example opt.resume arg: s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/logs/03-19-remote-GEN-353/checkpoints/xyz.ckpt
        logdir = "logs/" + opt.resume.split("/")[-3]
        run_id = logdir.split('-')[-1]
        run_name = 'GEN-' + run_id
        print(f"Resuming run {run_name}. Downloading the ckpt from S3 ({opt.resume}) to local ({logdir})")
        download_file(
            remote_path=opt.resume,
            local_path=logdir + "/checkpoints/last.ckpt",
        )

        print("Done")
        # Set the checkpoint to the last one in the logdir
        # resume_ckpt = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/logs/04-02-maclocal-GEN-412-test/checkpoints/end_epoch_1.ckpt" #Hardcoded for now
        resume_ckpt = logdir + "/checkpoints/last.ckpt"
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        date = datetime.datetime.now()
        date = date.strftime("%m-%d")
        logdir = f"logs/{date}-{opt.location}"

    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if "gpus" not in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        print("Attempting to load model ...")

        if opt.resume:
            # Remove all references to ckpts from the config:
            try:
                del config["model"]["params"]["first_stage_config"]["params"]["ckpt_path"]
            except Exception:
                pass
            try:
                del config["model"]["params"]["ckpt_path"]
            except Exception:
                pass
            try:
                del config["model"]["params"]["unet_config"]["params"]["ckpt_path"]
            except Exception:
                pass

                # Set the model ckpt to the last one from prev run.
            if opt.location in ['remote']:
                config.model.params.ckpt_path = resume_ckpt
                trainer_resume_ckpt = resume_ckpt
            elif opt.location in ['maclocal']:
                # Dit aanpassen als je local resumet
                config.model.params.ckpt_path = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/logs/04-02-maclocal-GEN-412-test/checkpoints/end_epoch_1.ckpt"
                trainer_resume_ckpt = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/logs/04-02-maclocal-GEN-412-test/checkpoints/end_epoch_1.ckpt"
        else:
            trainer_resume_ckpt = None

        if opt.location == "remote" and not opt.resume:
            print("Running remotely. Downloading pretrained models ...")

            try:
                remote_path = config.model.params.ckpt_path
                one_model_ckpt = True
            except Exception:
                one_model_ckpt = False

            if one_model_ckpt:
                download_file(
                    remote_path=config.model.params.ckpt_path,
                    local_path="/home/aiosyn/model.ckpt",
                )
                print("Downloaded one ckpt for the whole model. Ready to load.")
                config.model.params.ckpt_path = "/home/aiosyn/model.ckpt"
            else:
                print("Downloading UNET and first stage model ckpts separately...")
                try:
                    download_file(
                        remote_path=config.model.params.unet_config.params.ckpt_path,
                        local_path="/home/aiosyn/unet_model.ckpt",
                    )
                    config.model.params.unet_config.params.ckpt_path = "/home/aiosyn/unet_model.ckpt"
                except omegaconf.errors.ConfigAttributeError:
                    print("No unet checkpoint found, training from scratch")

                try:
                    download_file(
                        remote_path=config.model.params.first_stage_config.params.ckpt_path,
                        local_path="/home/aiosyn/first_stage_model.ckpt",
                    )
                    config.model.params.first_stage_config.params.ckpt_path = "/home/aiosyn/first_stage_model.ckpt"
                except omegaconf.errors.ConfigAttributeError:
                    print("No first stage checkpoint found")
        else:
            print("Models should already be downloaded.")

        model = instantiate_from_config(config.model)
        print("Model loaded.")

        print(f"Monitoring {model.monitor} for checkpoint metric.")

        # define my own trainer with id if resuming:
        if opt.resume:
            neptune_logger = NeptuneLogger(
                api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYTRjNmEyNy1lNTY5LTRmYTMtYjg5Yy03YjIxOTNhN2MwNGQifQ\=\=",
                project="generation",
                name=trainer_config["run_name"],
                log_model_checkpoints=trainer_config.log_model_checkpoints,
                with_id = run_name
            )
        else:
            neptune_logger = NeptuneLogger(
                api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYTRjNmEyNy1lNTY5LTRmYTMtYjg5Yy03YjIxOTNhN2MwNGQifQ\=\=",
                project="generation",
                name=trainer_config["run_name"],
                log_model_checkpoints=trainer_config.log_model_checkpoints,
            )
        # Log all hyperparams
        print(f"Monitoring {model.monitor} for checkpoint metric.")

        print(config)
        neptune_logger.log_hyperparams(config)
        neptune_logger.log_hyperparams(trainer_config)

        if not opt.resume:
            run_id = neptune_logger.experiment["sys/id"].fetch()
            logdir = logdir + f"-{run_id}"
        print(f"{logdir = }")

        ckptdir = os.path.join(logdir, "checkpoints")
        cfgdir = os.path.join(logdir, "configs")
        print(f"{ckptdir = }")
        checkpoint_callback = CustomModelCheckpoint(
            dirpath=ckptdir,
            save_top_k=1,
            save_last=True,
            monitor=model.monitor,
            save_weights_only=False,
        )
        config.data["location"] = opt.location

        print(f"Max epochs: {trainer_config.max_epochs}")
        print(f" {opt.resume = }")

        trainer = Trainer(
            max_epochs=trainer_config.max_epochs,
            accelerator="gpu",
            devices=1,
            logger=neptune_logger,
            callbacks=[checkpoint_callback, ThesisCallback()],
            resume_from_checkpoint=trainer_resume_ckpt,
            # num_sanity_val_steps=0, # DIT skipt de validation sanity check. Default = 2
            auto_scale_batch_size=True
        )

        ckptdir = os.path.join(logdir, "checkpoints")
        cfgdir = os.path.join(logdir, "configs")

        trainer.logger.experiment["location"] = opt.location
        trainer.logger.experiment["base"] = opt.base[0].split("/")[-1]

        assert config.data.location in [
            "local",
            "remote",
            "maclocal",
        ], "Data location should be 'local', 'maclocal, or 'remote'"
        # Add location to the data config params
        config.data["params"]["location"] = config.data.location
        if config.data.location in ["local", "maclocal"] and config.data.already_downloaded == True:
            print("Data already downloaded, skipping download ...")
        else:
            download_dataset(
                dataset_name=config.data.dataset_name,
                location=config.data.location
            )
        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint from melk.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                print("ckpt path:", ckpt_path)
                trainer.save_checkpoint(ckpt_path)
                sync_logdir(opt, trainer, logdir, overwrite=True)  # Sync logdir after training finishes

        import signal

        signal.signal(signal.SIGUSR1, melk)
        # run
        if opt.train:
            try:
                # Don't use! float32 is the way to go
                if trainer_config.get('autocast', False):
                    # Perform training with autocasting enabled.
                    with torch.amp.autocast(device_type='cuda'):
                        print("RUNNING WITH AUTOCAST ENABLED")
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
                print("Trainer has fitted the model.")
                sync_logdir(opt, trainer, logdir, overwrite=True)  # Sync logdir after training finishes
                print(f"Best model path: {checkpoint_callback.best_model_path}")
                print(f"Best model score: {checkpoint_callback.best_model_score}")
                trainer.logger.experiment["Best model path"] = checkpoint_callback.best_model_path
                trainer.logger.experiment["Best model score"] = checkpoint_callback.best_model_score
            except Exception:
                melk()
                raise
        elif not trainer_config.skip_validation:
            print("Skipped training.")
            trainer.validate(model, data)
            print("Done validating!")

    except Exception:
        # try:
        #     import pudb as debugger
        # except ImportError:
        #     import pdb as debugger
        # debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
