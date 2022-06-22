import os
import json

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDataset(torch.utils.data.Dataset):

    def __init__(self, imagepath, instancepath) -> None:
        super().__init__()

        self.instances = json.load(open(instancepath, "r"))

        self.images = [os.path.join(imagepath, i["file_name"]) for i in self.instances["images"]]
        self.bboxes, self.classes = self.get_bboxes()

        self.n_classes = np.asarray(self.classes).flatten().max()[0] + 1

    def resize_image(self, img_arr, bboxes, h, w):
        
        orig_h, orig_w = img_arr.shape[0], img_arr.shape[1]

        y_scale = h / orig_h
        x_scale = w / orig_w

        img = np.asarray(cv2.resize(img_arr, (h, w)))
        
        new_bboxes = np.ndarray((bboxes.shape))

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            x1_ = int(np.round(x1 * x_scale))
            y1_ = int(np.round(y1 * y_scale))
            x2_ = int(np.round(x2 * x_scale))
            y2_ = int(np.round(y2 * y_scale))
            new_bboxes[i] = np.asarray([x1_, y1_, x2_ - x1_, y2_ - y1_, bbox[4]])
        
        return {"image": img, "bboxes": new_bboxes}

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
        
        img = plt.imread(self.images[idx])
        bbox = self.bboxes[idx]
        classes = self.classes[idx]

        # Resize to 480, 640 - because it is the wished upon size

        bboxes = torch.zeros((len(classes), 6))

        for i in range(len(classes)):
            bboxes[i][0] = i

        for i, c in enumerate(classes):
            bbox[i].append(c)
        
        bbox = np.asarray(bbox)

        resized = self.resize_image(img, bbox, 512, 512)
        
        for i, r in enumerate(resized["bboxes"]):
            bboxes[i][1:] = torch.from_numpy(np.asarray(r))

        img = torch.tensor(resized["image"])
        bbox = bboxes
        idx = torch.tensor(idx)

        return {"img": img, "bbox": bbox, "id": idx}

    def visualize_annotations(self, idx):
        item = self[idx]

        bbox = item["bbox"]
        img = item["img"]

        _, ax = plt.subplots()

        ax.imshow(img)

        for bb in bbox:
            rect = patches.Rectangle((bb[1], bb[2]), bb[3], bb[4], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.show()

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        ids = list()

        for b in batch:
            images.append(b["img"])
            boxes.append(b["bbox"])
            ids.append(b["id"])

        images = torch.stack(images, dim=0)
        boxes = torch.cat(boxes, 0)

        return {"imgs": images, "bboxs": boxes, "ids": ids} # tensor (N, 3, 300, 300), 3 lists of N tensors each

if __name__ == "__main__":

    loader = ObjectDataset("data/val", "data/combined_val.json")

    print(len(loader))

    loader.visualize_annotations(4)