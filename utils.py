import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import config

def IOU(boxes_preds, boxes_labels, format="midpoints"):

    if format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    if format == "midpoints":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1)) + abs(
        (box2_x2 - box2_x1) * (box2_y2 - box2_y1)) - intersection

    return intersection / union


def NMS(bbs, iou_threshold, prob_threshold, format="corners"):
    assert type(bbs) == list
    bbs = [box for box in bbs if box[1] > prob_threshold]  # delete uncertain bounding boxes
    bbs = sorted(bbs, key=lambda x: x[1], reverse=True)
    bbs_after_nms = []

    while bbs:
        chosen_box = bbs.pop(0)
        bbs = [box for box in bbs if box[0] != chosen_box[0] or IOU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]),
                                                                    format) < iou_threshold]  # keep the different class boxes and compare the box with the same class
        bbs_after_nms.append(chosen_box)

    return bbs_after_nms


def MAP(pred_boxes, true_boxes, iou_threshold=0.5, format="corners", num_classes=20):
    AP = []
    epsilon = 1e-6
    # pred_boxes = [[train_idx,class, prob,x1,y1,x2,y2],[...],[...]]
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for ground_truth in true_boxes:
            if ground_truth[1] == c:
                ground_truths.append(ground_truth)

        amount_bbs = Counter([gt[0] for gt in ground_truths])  # img 0 has 3 bb, img 1 has 5 bb =>amount_bbs = {0:3,1:5}
        for key, val in amount_bbs.items():
            amount_bbs[key] = torch.zeros(val)  # amount_bbs = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bb = len(ground_truths)

        for idx, detection in enumerate(detections):
            best_iou = 0
            best_gts_idx = 0
            ground_truth_img = [bbs for bbs in ground_truths if bbs[0] == detection[0]]  # take bbs with pred in same img
            num_gts = len(ground_truth_img)
            for ii, gt in enumerate(ground_truth_img):
                iou = IOU(torch.tensor(detection[3:]), torch.tensor(gt[3:]), format=format)

                if iou > best_iou:
                    best_iou = iou
                    best_gts_idx = ii

            if (best_iou > iou_threshold) and amount_bbs[detection[0]][best_gts_idx] == 0:
                TP[idx] = 1
                amount_bbs[detection[0]][best_gts_idx] = 1
            else:
                FP[idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bb + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        AP.append(torch.trapz(precisions, recalls))

    return sum(AP) / len(AP)




def plot_image(image, boxes):
    class_label = config.COCO_LABELS if config.DATASET == 'COCO' else config.PASCAL_CLASSES
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_label))]
    print(boxes)
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        predicted_class = int(box[0])
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor = colors[predicted_class],
            facecolor = "none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_label[int(predicted_class)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(predicted_class)], "pad": 0},
        )

    plt.show()

def get_bboxes(
    loader,
    model,
    iou_threshold,
    prob_threshold,
    pred_format="cells",
    format ="midpoints",
    device="cuda",
    eva = False
):
    all_pred_boxes = []
    all_true_boxes = []


    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = NMS(
                bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
                format=format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def plot_some_images(number_of_images, model, loader, iou_threshold, prob_threshold):
    model.eval()
    for x, y in loader:
     x = x.to("cuda")
     for idx in range(number_of_images):
       bboxes = cellboxes_to_boxes(model(x))
       bboxes = NMS(bboxes[idx], iou_threshold=0.5, prob_threshold=0.4, format="midpoints")
       plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

    model.train()



def convert_cellboxes(predictions, grids=7):


    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / grids * (best_boxes[..., :1] + cell_indices)
    y = 1 / grids * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / grids * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(predictions,anchors, grid, is_preds=True):
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(grid)
            .repeat(predictions.shape[0], 3, grid, 1)
            .unsqueeze(-1)
            .to(predictions.device)
    )
    x = 1 / grid * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / grid * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / grid * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * grid * grid, 6)
    return converted_bboxes.tolist()

def save_checkpoint(state, filename="yolov1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def iou_width_height(boxes1, boxes2):

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union