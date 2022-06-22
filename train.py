from math import hypot
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import os
import yaml
import argparse

from models.yolos.yolo import Model as YoloModel
from models.yolos.google_utils import attempt_download 
from models.yolos.torch_utils import intersect_dicts
from models.yolos.general_utils import compute_loss
import test

from loader.dataset import ObjectDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Object-Detection Training Loop')

    parser.add_argument('--bs', default=8, type=int, help='Training batch size')
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--cuda", default=False, help="Whether to use CUDA or not")
    parser.add_argument("--cfg", default="models/yolos/configs/yolov4-p5.yaml", help="What config to use for the model")
    parser.add_argument("--pretrained", default=None, help="Path to pretrained file")
    parser.add_argument("--lr", default=1e-3, help="Learning rate of the model")
    parser.add_argument("--hyp", default="models/yolos/configs/hyp.scratch.yaml")

    args = parser.parse_args()

    if args.cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    print("Args:", args)

    # Create datasets
    train_dataset = ObjectDataset("data/train", "data/combined_train.json")
    val_dataset = ObjectDataset("data/val", "data/combined_val.json")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=val_dataset.collate_fn)

    
    # Load in the model to be used

    with open(args.cfg) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    nc = data_dict["nc"]

    if args.pretrained:
        attempt_download(args.pretrained)  # download if not found locally
        
        ckpt = torch.load(args.weights, map_location=args.device)  # load checkpoint
        model = YoloModel(args.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(args.device)  # create
        exclude = ['anchor'] if args.cfg else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), args.pretrained))  # report
    else:
        model = YoloModel(args.cfg, ch=3, nc=nc).to(args.device)# create
        #model = model.to(memory_format=torch.channels_last)  # create

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader) # For loading hyps

    model.hyp = hyp

    for epoch in range(args.epochs):



        for batch in train_loader:
            
            model.train()


            imgs = batch["imgs"].to(args.device) / 255
            bboxs = batch["bboxs"]
            ids = batch["ids"]

            imgs = imgs.permute(0, 3, 1, 2)

            pred = model(imgs)

            loss, loss_items = compute_loss(pred, bboxs.to(args.device), model)  # scaled by batch_size

            loss.backward()

            optimizer.zero_grad()

            print("loss:", loss)

            model.eval()

            results, maps = test.test(val_loader,
                            device=args.device,
                            model=model)
            
            print(results)
            
            
