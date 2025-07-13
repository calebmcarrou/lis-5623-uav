#######################################################################################
# eda.py
# Exploratory Data Analysis
#######################################################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
import random
import torch
import torchvision

def eda():
    """ Do Exploratory Data Analysis
    """
    img_dir = 'data/dataset/images'
    lab_dir = 'data/dataset/labels'
    subsets = ["train", "val"]
    # create image list, will be a sample of image means
    imgdf = []
    labdf = []
    # now loop through images
    for subset in subsets:
        ipath = os.path.join(img_dir, subset)
        lpath = os.path.join(lab_dir, subset)
        sample = random.sample(os.listdir(ipath), k=200)
        # iterate all images for rgb info
        for idx, img in enumerate(sample):
            print('Reading Image: '+str(idx)+'/'+str(len(sample)), end='\r')
            if img.endswith('jpg'): # ensure only images
                image = cv2.imread(os.path.join(ipath, img))
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                samp = rgb.mean(axis=(0, 1)) # sample of image

                imgdf.append({
                        "subset": subset,
                        "r": samp[0]/255, # normalize for all channels
                        "g": samp[1]/255,
                        "b": samp[2]/255
                })
            else:
                pass
    # create polars dataframe
    imgdf = pl.DataFrame(imgdf)
    # plot histogram of rgb channels
    fig, axs = plt.subplots(1, 3)
    axs[0].hist(imgdf["r"].to_numpy(), color='red')
    axs[0].set_title('Red Channel')
    axs[1].hist(imgdf["g"].to_numpy(), color='green')
    axs[1].set_title('Green Channel')
    axs[2].hist(imgdf["b"].to_numpy(), color='blue')
    axs[2].set_title('Blue Channel')
    plt.savefig('data/hist.png')
    # get mean and std info
    rmu, rsd = imgdf["r"].mean(), imgdf["r"].std()
    gmu, gsd = imgdf["g"].mean(), imgdf["g"].std()
    bmu, bsd = imgdf["b"].mean(), imgdf["b"].std()
    print("Red Channel ~ N("+str(round(rmu, 3))+","+str(round(rsd, 3))+")")
    print("Green Channel ~ N("+str(round(gmu, 3))+","+str(round(gsd, 3))+")")
    print("Blue Channel ~ N("+str(round(bmu, 3))+","+str(round(bsd, 3))+")")

if __name__ == "__main__":
    eda()