import torch
from tqdm import tqdm
from utils import check_class_accuracy, get_bboxes, NMS, MAP, cellboxes_to_boxes, convert_cellboxes, load_checkpoint, \
    save_checkpoint, plot_some_images, loaders
from loss import Loss
import torch.optim as optim
import config
from model import YOLOV3

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    running_loss = []
    loop = tqdm(train_loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        # x = x.type(torch.FloatTensor)
        x = x.to(config.DEVICE)

        y1, y2, y3 = (y[0].to(config.DEVICE),
                      y[1].to(config.DEVICE),
                      y[2].to(config.DEVICE),
                      )
        with torch.cuda.amp.autocast():  # float16 faster
            out = model(x)
            loss = (
                    loss_fn(out[0], y1, scaled_anchors[0])
                    + loss_fn(out[1], y2, scaled_anchors[1])
                    + loss_fn(out[2], y3, scaled_anchors[2])
            )

        running_loss.append(loss.item())

        for param in model.parameters():  # optimize
            param.grad = None

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(running_loss) / len(running_loss)
        loop.set_postfix(loss=mean_loss)


def evaluation(test_loader, model, optimizer, file, lr, scaled_anchors):
    load_checkpoint(file, model, optimizer, lr)
    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                          prob_threshold=config.CONF_THRESHOLD, anchors=config.ANCHORS)
    mean_avg_pre = MAP(pred_boxes, target_boxes, iou_threshold=config.MAP_IOU_THRESH, format="midpoints")
    print(f"Evaluation mAP: {mean_avg_pre}")
    plot_some_images(model, test_loader, iou_threshold=config.NMS_IOU_THRESH, prob_threshold=config.CONF_THRESHOLD,
                     anchors=scaled_anchors)


def main():
    yolov3 = YOLOV3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(yolov3.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    loss = Loss()
    train_loader, test_loader, val_loader = loaders()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, yolov3, optimizer, config.LEARNING_RATE)

    # we need to scale the anchors to be relevant with cell by multiply the anchor width and height with grid size
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.GRIDS).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # same size for multiplication
    ).to(config.DEVICE)
    print("using: ", config.DEVICE)

    if config.EVALUATION:
        evaluation(test_loader, yolov3, optimizer, "yolov1.pth.tar", config.LEARNING_RATE, scaled_anchors)
    else:
        maps = 0.05
        for epoch in range(config.NUM_EPOCHS):
            train(train_loader, yolov3, optimizer, loss, scaler, scaled_anchors)
            if epoch > 0 and epoch % 3 == 0:
                check_class_accuracy(yolov3, test_loader, threshold=config.CONF_THRESHOLD)
                pred_boxes, target_boxes = get_bboxes(val_loader, yolov3,
                                                      iou_threshold=config.NMS_IOU_THRESH,
                                                      prob_threshold=config.CONF_THRESHOLD,
                                                      anchors=config.ANCHORS)

                mean_avg_pre = MAP(pred_boxes, target_boxes,
                                   iou_threshold=config.MAP_IOU_THRESH,
                                   format="midpoints",
                                   num_classes=config.NUM_CLASSES)
                if maps < mean_avg_pre:
                    save_checkpoint(yolov3, optimizer)
                    maps = mean_avg_pre

                print(f"Mean Average Precision for epoch {epoch}: {mean_avg_pre}")

        save_checkpoint(yolov3, optimizer,"FINAL.pth.tar")


if __name__ == "__main__":
    main()
