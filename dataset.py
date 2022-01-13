import torch
import pandas as pd
from PIL import Image, ImageFile
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import (iou_width_height as IOU, cellboxes_to_boxes, NMS,plot_image)
import config
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ObjectDetectionDataset(Dataset):
    def __init__(self, csv_file, image_path, label_path, anchors, grids=[13,26,52], num_class=20, transform=None):
        super(ObjectDetectionDataset, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.images_path = image_path
        self.labels_path = label_path
        self.transform = transform
        self.grids = grids
        self.num_class = num_class
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors//3
        self.ignore_iou_thresh = 0.5

    def __getitem__(self, index):
        image = np.array(Image.open(os.path.join(self.images_path, self.annotations.iloc[index, 0])).convert("RGB"))
        boxes = []
        lab_path = os.path.join(self.labels_path, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname = lab_path, delimiter = " ", ndmin = 2), -1, axis = 1).tolist()
        # format (x,y,w,h,class) for augmentations
        print(bboxes)
        if self.transform:
            augmentations = self.transform(image = image, bboxes = bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        label_matrix = [torch.zeros((self.num_anchors // 3, grid, grid, 6 )) for grid in self.grids] #6 included x ,y , w ,h , class , proobj
        # so the shape will be (3,3,grid,grid,6) => we have 3 scales 13x13, 26x26, 52x52
        # and each scale we have 3 anchor boxes predict each grid cell

        for box in bboxes:
            iou_anchors = IOU(torch.tensor(box[2:4]),self.anchors)
            anchor_indices = iou_anchors.argsort(descending = True, dim =0)
            x, y,width, height, class_label = box
            class_label = int(class_label)
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # 0, 1, 2 which scale this anchor belongs 13? 26? 52?
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # 0, 1, 2 which anchor in the calculated scale
                grid = self.grids[scale_idx]
                # convert to relative to the cell

                i, j = int(grid * x), int(grid * y)
                anchor_taken = label_matrix[scale_idx][anchor_on_scale, j, i, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    label_matrix[scale_idx][anchor_on_scale, j, i, 0] = 1
                    x_cell, y_cell = (grid*x - i), (grid*y - j)
                    w_cell, h_cell = (width*grid), (height*grid)
                    box_coords = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                    label_matrix[scale_idx][anchor_on_scale, j, i, 1:5] = box_coords
                    label_matrix[scale_idx][anchor_on_scale, j, i, 5] = class_label
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    label_matrix[scale_idx][anchor_on_scale, j, i, 0] = -1

        return image, tuple(label_matrix)

    def __len__(self):
        return len(self.annotations)


def test():
    anchors = config.ANCHORS

    transform = config.train_transforms

    dataset = ObjectDetectionDataset(
        config.CSV_PATH,
        config.IMG_DIR,
        config.LABEL_DIR,
        anchors=anchors,
        grids= [13, 26, 52],
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cellboxes_to_boxes(
                y[i], is_preds=False, grid=y[i].shape[2], anchors=anchor
            )[0]
        boxes = NMS(boxes, iou_threshold=1, prob_threshold=0.7, format="midpoints")
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)

test()