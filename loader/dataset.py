import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDataset(torch.utils.data.Dataset):

    def __init__(self, imagepath, instancepath) -> None:
        super().__init__()

        self.instances = json.load(open(instancepath, "r"))

        print(self.instances.keys())

        self.images = [os.path.join(imagepath, i["file_name"]) for i in self.instances["images"]]
        self.bboxes = self.get_bboxes()

    def get_bboxes(self):
        
        bboxes = []
        
        for image in self.instances["images"]:
            
            image_boxes = []
            
            image_id = image["id"]
            
            for annotation in self.instances["annotations"]:
                if annotation["image_id"] == image_id:
                    image_boxes.append(annotation["bbox"])
            
            bboxes.append(image_boxes)
        
        return bboxes

    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = plt.imread(self.images[idx])
        bbox = self.bboxes[idx]

        return img, bbox

    def visualize_annotations(self, idx):
        img, bbox = self[idx]

        print("Number of boxes:", len(bbox))

        fig, ax = plt.subplots()

        ax.imshow(img)

        for bb in bbox:
            rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        plt.show()

if __name__ == "__main__":

    loader = ObjectDataset("data/val", "data/combined_val.json")

    print(len(loader))

    loader.visualize_annotations(60)