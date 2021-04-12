import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image

if __name__ == '__main__':
    img_pth = './samples/861-0.jpg'
    img = Image.open(img_pth)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(1)])
    img_shear = transform(img)

    img = np.array(img)
    img_shear = np.array(img_shear)

    plt.figure(0)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_shear)

    plt.show()
