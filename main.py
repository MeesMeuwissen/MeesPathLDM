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
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from get_data import download_dataset
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config, plot_images
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
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size : (worker_id + 1) * split_size]
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
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        trainer.logger.log_metrics({"train/Loss": outputs["loss"]}, step=trainer.global_step)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("THESISCALLBACK: Training is starting...")

        # Log the run location to metadata directly
        trainer.logger.experiment["Location"] = opt.location

    def on_train_end(self, trainer, pl_module):
        print("Training completed.")

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        split = "train"
        # log the batch every 500 batches ?
        if batch_idx % 500 == 0:
            plot_images(trainer, batch["image"], batch_idx, 4, len(batch["image"]) // 4, split=split)
        # print(f"Logged the batch to batch_samples/{split}/{batch_idx}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Potentially fun to sample a single image after every, say, 10 epochs with the same caption to see it
        # hopefully progress
        print(f"Finished the epoch, global step:{trainer.global_step}. Syncing logdir ...")

        #Sync the whole logdirectory with aws, so upload it and overwrite is ok
        set_sso_profile("aws-aiosyn-data", region_name="eu-west-1")
        upload_directory(logdir, f"s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/{logdir}", overwrite=True)
        print("Done syncing logdir.")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # print(f"Starting epoch {trainer.current_epoch}. ")
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Starting validation ...")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        samples = trainer.model.validation_step_outputs[
            0
        ]  # A batch of 8 imgs I think, it has shape (8,3,256,256), dtype uint8)
        samples_t = torch.from_numpy(samples / 255.0)

        grid = torchvision.utils.make_grid(samples_t, nrow=4, padding=10)
        grid = transforms.functional.to_pil_image(grid)

        trainer.logger.experiment[f"validation_samples/epoch_{trainer.current_epoch}"].log(
            grid,
            name=f"Validation sample created at epoch {trainer.current_epoch} and step {trainer.global_step}.",
            description=f'Caption used: "A H&E stained slide of a piece of kidney tissue" (If I did not forget to change this at least...',
        )
        print("Logged the validation images!")


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
    torch.set_float32_matmul_precision("medium")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    sys.path.append(os.getcwd())
    if opt.location in ["local", "maclocal"]:
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

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
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
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
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

        if opt.location == "remote":
            print("Running remotely. Downloading pretrained models ...")

            download_file(
                remote_path="s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/unet/pathldm/epoch_3-001.ckpt",
                local_path="/home/aiosyn/code/generationLDM/pretrained/srikar/epoch_3-001.ckpt",
            )
            print("Downloaded unet. Ready to load.")
        else:
            print("Models should already be downloaded.")

        model = instantiate_from_config(config.model)
        print("Model loaded.")

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "neptune": {
                "target": "pytorch_lightning.loggers.NeptuneLogger",
                "params": {
                    "name": "neptune",
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["neptune"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:01}",
                "verbose": True,
                "save_last": True,
            },
        }
        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "cuda_callback": {"target": "main.CUDACallback"},
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})
        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_opt.resume_from_checkpoint
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        print(f"Monitoring {model.monitor} for checkpoint metric.")
        checkpoint_callback = ModelCheckpoint(
            dirpath=logdir,
            save_top_k=1,
            save_last=True,
            monitor=model.monitor,
            save_weights_only=trainer_config["weights_only"],
        )

        # define my own trainer:
        neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyYTRjNmEyNy1lNTY5LTRmYTMtYjg5Yy03YjIxOTNhN2MwNGQifQ\=\=",
            project="generation",
            name=trainer_config["run_name"],
            log_model_checkpoints=trainer_config.log_model_checkpoints,
        )
        # Log all hyperparams
        neptune_logger.log_hyperparams(params=config)
        neptune_logger.log_hyperparams(params=trainer_config)

        print(f"Max epochs: {trainer_config.max_epochs}")

        trainer = Trainer(
            max_epochs=trainer_config.max_epochs,
            accelerator="gpu",
            devices=1,
            logger=neptune_logger,
            callbacks=[ThesisCallback(), checkpoint_callback],
            default_root_dir="s3://aiosyn-data-eu-west-1-bucket-ops/models/generation/logs/"
        )
        #trainer.logdir = logdir  ###
        print(f"{logdir = }")
        config.data["location"] = opt.location
        trainer.logger.experiment["Location"] = opt.location

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
                location=config.data.location,
                subsample=config.data.params.train.params.config.subsample,
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
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                print("ckpt path:", ckpt_path)
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        # run
        if opt.train:
            try:
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):  # This apparently solves everything
                    if trainer_config.skip_validation:
                        # Only perform the training, no FID computation.
                        trainer.fit(model, data, val_dataloaders=None)
                    else:
                        trainer.fit(model, data)
                    print("Trainer has fitted the model.")
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
