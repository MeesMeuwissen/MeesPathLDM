import matplotlib.pyplot as plt
import numpy as np
import PIL
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

#find mnist
import os
os.chdir("/mnt/c/Users/MeesMeuwissen/Documents/Aiosyn/code/ThesisProject")

mnist = datasets.MNIST(root="./MNISTdata")

transform = transforms.Compose([transforms.Resize((256, 256))])

img = mnist[0][0]
print(img)

img = transform(img)
print(img)

img = img.convert("RGB")
print(img)
print("SUCCESS")

plt.imshow(img)
plt.show()

tens = transforms.ToTensor()(img).permute(1,2,0)
print(tens.shape)
print(f"Neptune mode:{os.environ['$NEPTUNE_MODE']}")
