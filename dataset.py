import torch
import pandas as pd
from PIL import Image, ImageFile
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import (iou_width_height as IOU)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ObjectDetectionDataset(Dataset):
    def __init__(self, csv_file, image_path, label_path, anchors, grids=[13,26,52], num_class=20, num_boxes=2, transform=None):
        super(ObjectDetectionDataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.images_path = image_path
        self.labels_path = label_path
        self.transform = transform
        self.grids = grids
        self.num_class = num_class
        self.num_boxes = num_boxes

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.images_path, self.annotations.iloc[index, 0]))
        boxes = []
        fl = open(os.path.join(self.labels_path, self.annotations.iloc[index, 1]), 'r')
        data = fl.read().splitlines()
        fl.close()

        for idx, lab in enumerate(data):
            data[idx] = lab.strip()
            c, x, y, w, h = map(float, data[idx].split(' '))
            coords = [int(c), x, y, w, h]
            boxes.append(coords)
        boxes = torch.tensor(boxes)
        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.grids, self.grids,
                                    self.num_class + 5 * self.num_boxes))  # target = [grids, grids, 25] with 20 class and 5 for the bounding box
        for box in boxes:
            c, x, y, w, h = box.tolist()
            c = int(c)
            i, j = int(self.grids * x), int(self.grids * y)
            x_cell, y_cell = (self.grids * x - i), (
                        self.grids * y - j)  # normally x,y,w,h relative to the full image so we have to convert it relative to cell
            w_cell, h_cell = (w * self.grids), (h * self.grids)
            if label_matrix[j, i, 20] == 0:
                label_matrix[j, i, 20] = 1
                box_coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[j, i, 21:25] = box_coords
                label_matrix[j, i, c] = 1

        return image, label_matrix

    def __len__(self):
        return len(self.annotations)
