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

args = get_parser().parse_args()
setup_seed(args.seed)

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
valid_dataset = VOC_Dataset(desc_dict, args.valid_img_path, args.valid_mask_path, img_size=args.img_size)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

print("Testing set volumn: {}".format(len(valid_dataset)))

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

checkpoints = torch.load(args.checkpoint_path)
net.load_state_dict(checkpoints["model_state_dict"])
net.eval()

valid_dice = []
valid_hd95 = []
valid_f1 = []
valid_iou = []

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

        dice, hd95, miou, f1_score = accuracy(torch.argmax(logits.clone(), dim=1), mask.clone())
        valid_dice.extend(dice)
        valid_hd95.extend(hd95)
        valid_f1.extend(f1_score)
        valid_iou.extend(miou)

valid_dice = np.mean(np.array(valid_dice))
valid_hd95 = np.mean(np.array(valid_hd95))
valid_f1 = np.mean(np.array(valid_f1))
valid_iou = np.mean(np.array(valid_iou))
print("Dice: {:.4f} HD95: {:.4f} F1: {:.4f} IoU: {:.4f}"\
        .format(valid_dice, valid_hd95, valid_f1, valid_iou))