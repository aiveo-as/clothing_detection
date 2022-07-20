import os
import json
from matplotlib import image

from tqdm import tqdm

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDataset(torch.utils.data.Dataset):

    def __init__(self, imagepath, instancepath, img_dim=[512, 512]) -> None:
        super().__init__()

        self.ydim, self.xdim = img_dim[0], img_dim[1]

        self.instances = json.load(open(instancepath, "r"))

        self.images = [os.path.join(imagepath, i["file_name"]) for i in self.instances["images"]]
        self.bboxes, self.classes = self.get_bboxes()

        self.n_classes = np.asarray(self.classes).flatten().max()[0] + 1

        self.shapes = self.get_shapes()

        print("self.shapes.shape:", self.shapes.shape)

        self.labels = self.get_labels()

    
    def get_shapes(self):
        
        shapes = np.zeros((len(self.images), 2))

        for idx, im_path in enumerate(self.images):
            im = plt.imread(im_path)
            shapes[idx] = im.shape[:2]

        return shapes

    def get_labels(self):
        
        labels = list()

        for _, imagex in tqdm(enumerate(range(len(self.classes))), total=len(self.classes), desc="Loading labels"):
            per_image_labels = np.zeros((len(self.classes[imagex]), 6))
            for idx in range(len(self.classes[imagex])):
                per_image_labels[idx][1] = self.classes[imagex][idx]
                bbox = self.bboxes[imagex][idx]
                y, x = self.shapes[imagex]
                bbox[0], bbox[2] = bbox[0] / x, bbox[2] / x
                bbox[1], bbox[3] = bbox[1] / y, bbox[3] / y
                per_image_labels[idx][2:] = bbox

            labels.append(per_image_labels)
        
        return labels

    def get_bboxes(self):
        
        bboxes = []
        classes = []
        
        for image in self.instances["images"]:
            
            image_boxes = []
            image_classes = []
            
            image_id = image["id"]
            
            for annotation in self.instances["annotations"]:
                if annotation["image_id"] == image_id:
                    image_boxes.append(annotation["bbox"])
                    image_classes.append(annotation["category_id"])
            
            bboxes.append(image_boxes)
            classes.append(image_classes)

        return bboxes, classes

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = cv2.imread(self.images[idx])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.xdim, self.ydim))

        labels = self.labels[idx]

        img = torch.tensor(img).permute(2, 0, 1)
        bbox = torch.tensor(labels)

        return img, bbox, self.images[idx], None

    def visualize_annotations(self, idx):
        item = self[idx]

        bbox = item[1]
        img = item[0].permute(1, 2, 0)

        _, ax = plt.subplots()

        print("bbox:", bbox)

        ax.imshow(img)

        for bb in bbox:
            rect = patches.Rectangle((bb[1] * self.xdim, bb[2] * self.ydim), bb[3] * self.xdim, bb[4] * self.ydim, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.savefig("test_viz.png")

    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        paths = list()
        shapes = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            paths.append(b[2])
            shapes.append(b[3])
        
        for i, l in enumerate(boxes):
            l[:, 0] = i

        images = torch.stack(images, dim=0)
        boxes = torch.cat(boxes, 0)

        return images, boxes, paths, shapes # tensor (N, 3, 300, 300), 3 lists of N tensors each

if __name__ == "__main__":

    loader = ObjectDataset("data/val", "data/combined_val.json")

    print(len(loader))

    print(loader[0])