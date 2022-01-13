import torch
from tqdm import tqdm
from utils import get_bboxes, NMS, MAP, cellboxes_to_boxes, convert_cellboxes, load_checkpoint, save_checkpoint, plot_some_images, loaders
from loss import Loss
import torch.optim as optim
import config
from model import YOLOV3
torch.backends.cudnn.benchmark = True


def train(train_loader, model, optimizer, loss, scaler, scaled_anchors):
    running_loss = []
    loop = tqdm(train_loader, leave=True)
    for idx,(x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y1, y2, y3 = ( y[0].to(config.DEVICE),
                       y[1].to(config.DEVICE),
                       y[2].to(config.DEVICE),
                       )
        with torch.cuda.amp.autocast(): #float16 faster
            out = model(x)
            loss = (
                    loss(out[0], y1, scaled_anchors[0])
                    + loss(out[1], y2, scaled_anchors[1])
                    + loss(out[2], y3, scaled_anchors[2])
            )

        running_loss.append(loss.item())
        for param in model.parameters(): #optimize
            param.grad = None

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(running_loss) / len(running_loss)
        loop.set_postfix(loss=mean_loss)


def main():
    YOLOv3 = YOLOV3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(YOLOv3.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()
    loss = Loss()
    train_loader, test_loader, train_eval_loader = loaders()

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(file), model, optimizer)