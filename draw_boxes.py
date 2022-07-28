import cv2 as cv
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

if __name__ == "__main__":

    imgpaths = "data/images/val"
    labelpaths = "data/labels/val"

    imgfile = os.listdir(imgpaths)[0]

    labelfile = imgfile.split(".")[0] + ".txt"

    bboxes = []

    with open(os.path.join(labelpaths, labelfile), "r") as f:
        for l in f.readlines():
            t = l.split(" ")
            t = [float(x) for x in t]
            bboxes.append(t)

    img = cv.imread(os.path.join(imgpaths, imgfile))

    img = cv.resize(img, (700, 300), cv.INTER_AREA)

    h, w = img.shape[:2]

    print(h, w)

    _, ax = plt.subplots()

    ax.imshow(img)

    for bb in bboxes:
        rect = patches.Rectangle((bb[1] * w, bb[2] * h), bb[3] * w, bb[4] * h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.savefig("test.jpg")