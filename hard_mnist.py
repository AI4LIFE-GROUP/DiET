import torch
import pdb
import copy
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import glob
import time
import random


def construct_data(path, out_path):

    small_resize = transforms.Resize(56)
    resize = transforms.Resize(112)
    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    files = glob.glob(path)

    for i, f in enumerate(files):
        if f[-3:] != "png":
            continue
        
        label = int(f.split("/")[-2])
        image = Image.open(f)
        t = to_tensor(resize(image))

        # create background
        new_tensor = torch.ones((3,224,224)).to(t.dtype)*t[:,0:1,0:1]

        # add noise (lines)
        num_lines = torch.randint(low=0, high=3, size=(2,))
        for i in range(num_lines[0]):
            width = random.randint(0, 50)
            ind = random.randint(0, 223-(width+1))
            new_tensor[:, ind:ind+width, :] = torch.rand(3, width, 224)
        for i in range(num_lines[1]):
            width = random.randint(0, 50)
            ind = random.randint(0, 223-(width+1))
            new_tensor[:, :, ind:ind+width] = torch.rand(3, 224, width)


        # add a small number somewhere
        f2 = files[random.randint(0, len(files)-1)]
        while (f2[-3:] != "png"):
            f2 = files[random.randint(0, len(files)-1)]
        image2 = Image.open(f2)
        t2 = to_tensor(small_resize(image2))
        ind2 = torch.randint(low=0, high=164, size=(2,))
        small_digit_background = torch.where(torch.sum(t2, 0)<=2, 0, 1)
        t2 *= small_digit_background
        new_tensor[:, ind2[0]:ind2[0]+56, ind2[1]:ind2[1]+56] += (0.98*t2)

        # actual/main number
        ind = torch.randint(low=0, high=112, size=(2,))
        # main_digit_background = torch.where(torch.sum(t, 0)<=2, 0, 1)
        # t *= main_digit_background
        new_tensor[:, ind[0]:ind[0]+112, ind[1]:ind[1]+112] = t
        new_tensor.clamp(max=1, min=0)

        to_image(new_tensor).save(out_path + "/" + str(label) + "/" + f.split("/")[-1])

    return 


def main():

    construct_data("data/colorized-MNIST/training/*/*", "data/hard_mnist/training/")
    construct_data("data/colorized-MNIST/testing/*/*", "data/hard_mnist/testing/")
    print("constructed hard MNIST dataset")

    
    
if __name__ == "__main__":
    main()


