import logging
import os
import random
import numpy as np
import torch
import clip
import time
import warnings
warnings.filterwarnings("ignore")

from networks.vision_transformer import SwinUnet
from tools.get_parser import get_parser
from utils import *
from dataloader import VOC_Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter 

args = get_parser().parse_args()
setup_seed(args.seed)

## set up logger
path = "logs/{}_{}".format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), args.model_name)
args.saved_path = path
os.makedirs(path)
log_file_path = os.path.join(path, "log.txt")
logger = get_logger(log_file_path)
logger.info("{}".format(log_file_path))
logger.info("{}".format(args))

## set up tensorboard writer
writer = SummaryWriter(path)

## load clip model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _ = clip.load("ViT-L/14@336px", device=device)
clip_model.eval()
vision_encoder = clip_model.visual
for name, param in vision_encoder.named_parameters():
    param.requires_grad = False

## load description features
desc_dict = make_desc_features(args.description_path, clip_model, device)

## load dataset
train_dataset = VOC_Dataset(desc_dict, args.train_img_path, args.train_mask_path, img_size=args.img_size)
valid_dataset = VOC_Dataset(desc_dict, args.valid_img_path, args.valid_mask_path, img_size=args.img_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))

desc_feat = torch.tensor([desc_dict[k]["features"] for k in desc_dict.keys()]).to(device, dtype=torch.float32)   # (n_classes, 5, 768)
class_desc_feat = torch.tensor([desc_dict[k]["description"] for k in desc_dict.keys()]).to(device, dtype=torch.float32)   # (n_classes, 768)
## define model
net = SwinUnet(img_size=args.img_size, 
                embed_dim=args.embed_dim,
                depths_enoder=args.depths_encoder,
                depths_decoder=args.depths_decoder,
                num_heads=args.num_heads,
                num_classes=args.num_classes,
                dropout=args.dropout).to(device)

dice_loss = DiceLoss(n_classes=args.num_classes)
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr)

## train
min_val_iou = 0x7fffffff

for epoch in range(args.epochs):

    train_dice = []
    train_hd95 = []
    train_f1 = []
    train_iou = []
    train_loss = 0

    net.train()
    for i, (cur_class, img, mask) in enumerate(train_loader):
        '''
        cur_class: (batch_size)
        img: (batch_size, 3, img_size, img_size)
        mask: (batch_size, img_size, img_size)
        '''
        img = img.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.long)
        cur_class = cur_class.to(device, dtype=torch.long)

        if args.contrast == "False":
            logits, img_text_logits = net(img, desc_feat.clone().detach()) 
        elif args.contrast == "feature":
            logits, img_text_logits = net(img, desc_feat.clone().detach()) 
        elif args.contrast == "description":
            logits, img_text_logits = net(img, class_desc_feat.clone().detach())

        loss_dice = dice_loss(logits, mask, softmax=True)
        loss_ce = ce_loss(logits, mask)
        loss = 0.6*loss_dice + 0.4*loss_ce

        if args.contrast != "False":
            contrast_loss = sum([ce_loss(img_text_logits[i], cur_class) 
                                for i in range(len(img_text_logits))]) / len(img_text_logits)
            loss = loss + contrast_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        dice, hd95, miou, f1_score = accuracy(torch.argmax(logits.clone(), dim=1), mask.clone())
        train_dice.extend(dice)
        train_hd95.extend(hd95)
        train_f1.extend(f1_score)
        train_iou.extend(miou)

    train_loss /= len(train_loader)
    train_dice = np.mean(np.array(train_dice))
    train_hd95 = np.mean(np.array(train_hd95))
    train_f1 = np.mean(np.array(train_f1))
    train_iou = np.mean(np.array(train_iou))
    logger.info("Epoch: {} Training Loss: {:.4f} Dice: {:.4f} HD95: {:.4f} F1: {:.4f} IoU: {:.4f}"\
                .format(epoch, train_loss, train_dice, train_hd95, train_f1, train_iou))

    valid_loss = 0
    valid_dice = []
    valid_hd95 = []
    valid_f1 = []
    valid_iou = []

    net.eval()
    with torch.no_grad():
        for i, (cur_class, img, mask) in enumerate(valid_loader):
            img = img.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            cur_class = cur_class.to(device, dtype=torch.long)

            if args.contrast == "False":
                logits, img_text_logits = net(img, desc_feat.clone().detach()) 
            elif args.contrast == "feature":
                logits, img_text_logits = net(img, desc_feat.clone().detach()) 
            elif args.contrast == "description":
                logits, img_text_logits = net(img, class_desc_feat.clone().detach())

            loss_dice = dice_loss(logits, mask, softmax=True)
            loss_ce = ce_loss(logits, mask)
            loss = 0.6*loss_dice + 0.4*loss_ce

            if args.contrast != "False":
                contrast_loss = sum([ce_loss(img_text_logits[i], cur_class) 
                                    for i in range(len(img_text_logits))]) / len(img_text_logits)
                loss = loss + contrast_loss

            valid_loss += loss.item()
            dice, hd95, miou, f1_score = accuracy(torch.argmax(logits.clone(), dim=1), mask.clone())
    
            valid_dice.extend(dice)
            valid_hd95.extend(hd95)
            valid_f1.extend(f1_score)
            valid_iou.extend(miou)
    
    valid_loss /= len(valid_loader)
    valid_dice = np.mean(np.array(valid_dice))
    valid_hd95 = np.mean(np.array(valid_hd95))
    valid_f1 = np.mean(np.array(valid_f1))
    valid_iou = np.mean(np.array(valid_iou))
    logger.info("Epoch: {} Validation Loss: {:.4f} Dice: {:.4f} HD95: {:.4f} F1: {:.4f} IoU: {:.4f}"\
                .format(epoch, valid_loss, valid_dice, valid_hd95, valid_f1, valid_iou))
    
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/dice", train_dice, epoch)
    writer.add_scalar("train/hd95", train_hd95, epoch)
    writer.add_scalar("train/f1_score", train_f1, epoch)
    writer.add_scalar("train/iou", train_iou, epoch)

    writer.add_scalar("valid/loss", valid_loss, epoch)
    writer.add_scalar("valid/dice", valid_dice, epoch)
    writer.add_scalar("valid/hd95", valid_hd95, epoch)
    writer.add_scalar("valid/f1_score", valid_f1, epoch)
    writer.add_scalar("valid/iou", valid_iou, epoch)

    if valid_iou < min_val_iou:
        min_val_iou = valid_iou
        torch.save(net.state_dict(), 
                   os.path.join(path, "{}_dice_{}_hd95_{}_f1_{}_iou_{}.pth"\
                                .format(args.model_name, valid_dice, valid_hd95, valid_f1, valid_iou))
        )
    