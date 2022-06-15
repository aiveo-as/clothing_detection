import csv
import os
import json

from tqdm import tqdm

if __name__ == "__main__":

    # The bboxes in modanet is: [x1, y1, dx, dy] - Which should be transformed to [x1, y1, x2, y2]

    modanet = json.load(open("data/modanet/instances_all_modanet.json", "r"))

    for _, annotation in tqdm(enumerate(modanet["annotations"]), total=len(modanet["annotations"])):
        bb = annotation["bbox"]
        annotation["bbox"] = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
    
    json.dump(modanet, open("data/modanet/instances_all_modanet.json", "w"))

