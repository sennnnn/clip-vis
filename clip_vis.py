import argparse
import random

import clip
import cv2
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def apply_seaborn_colormap(im_gray, cm="rocket"):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # decode sns palette
    cmap = np.array(sns.color_palette(cm))
    cmap = (cmap * 255).astype(np.uint8)
    cmap = torch.tensor(cmap).permute(1, 0)[None].float()
    cmap = F.interpolate(cmap, 256, mode="linear")[0].cpu().numpy().astype(np.uint8)

    # Red, Green, Blue
    lut[:, 0, 0] = cmap[0]
    lut[:, 0, 1] = cmap[1]
    lut[:, 0, 2] = cmap[2]

    # Apply custom colormap through LUT
    im_color = cv2.LUT(cv2.cvtColor(im_gray, cv2.COLOR_GRAY2RGB), lut)

    return im_color


def local_blend(main_img, sub_img, mask, alpha=0.5):
    main_img[mask > 0, 0] = alpha * main_img[mask > 0, 0] + (1 - alpha) * sub_img[mask > 0, 0]
    main_img[mask > 0, 1] = alpha * main_img[mask > 0, 1] + (1 - alpha) * sub_img[mask > 0, 1]
    main_img[mask > 0, 2] = alpha * main_img[mask > 0, 2] + (1 - alpha) * sub_img[mask > 0, 2]

    return main_img


def global_blend(main_img, sub_img, alpha=0.5):
    return cv2.addWeighted(main_img, alpha, sub_img, 1 - alpha, 1.0)


class CLIPVisual(nn.Module):

    def __init__(self, model_name="ViT-B/16", device="cuda"):
        # compatible with python2.x
        super(CLIPVisual, self).__init__()
        assert model_name in ["ViT-B/32", "ViT-B/16"], "only support vit based clip model."

        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image)

    def encode_spatial_image(self, image):
        vis_model = self.model.visual
        dtype = self.model.dtype

        # convert pil to tensor
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        x = x.type(dtype)

        # stem conv
        x = vis_model.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gridH, gridW = x.shape

        # before transformer
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            vis_model.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vis_model.positional_embedding.to(x.dtype)
        x = vis_model.ln_pre(x)

        # transformer forward
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = vis_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # after transformer
        # x[:, 0, :] = vis_model.ln_post(x[:, 0, :])

        # make spatial tokens
        cls_token = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        patch_tokens = patch_tokens.reshape(b, gridH, gridW, patch_tokens.shape[-1])

        return cls_token, patch_tokens


model = CLIPVisual()


def top_percen_sim_tokens_vis(img, cls_token, patch_tokens, rate=0.25, palette="mako"):

    H, W = img.shape[:2]
    gridH, gridW = patch_tokens.shape[0], patch_tokens.shape[1]
    patch_tokens = patch_tokens.flatten(0, 1)  # HWC -> NC

    # norm
    # cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
    # patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

    # l2 norm
    cls_token = F.normalize(cls_token, p=2, dim=-1)
    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

    num_tokens = patch_tokens.shape[0]

    # cal sim
    sim_score = torch.einsum("c,nc->n", cls_token, patch_tokens)

    # get top (rate) percentage patch tokens mask
    sort_ids = torch.argsort(sim_score, dim=0, descending=True)
    truncate_num = int(num_tokens * rate)
    truncate_ids = sort_ids[:truncate_num]
    truncate_mask = torch.zeros_like(sim_score)
    truncate_mask[truncate_ids] = 1

    # make spatial score & mask
    truncate_mask = truncate_mask.reshape(gridH, gridW)
    sim_score = sim_score.reshape(gridH, gridW)

    score = F.interpolate(sim_score[None, None], size=(H, W), mode="bilinear")[0, 0].cpu().numpy().astype(np.float32)
    mask = F.interpolate(truncate_mask[None, None], size=(H, W), mode="nearest")[0, 0].cpu().numpy().astype(np.float32)

    # use expoential axis
    score = np.exp(score)
    norm_score = (score - np.min(score)) / (np.max(score) - np.min(score))

    # make color sim score
    color_score = apply_seaborn_colormap((norm_score * 255).astype(np.uint8), palette)

    return color_score, mask


def parse_args():
    parser = argparse.ArgumentParser("visualize relationship between cls token and patch tokens.")
    parser.add_argument("-i", "--path", default="demo.jpg", type=str, help="raw image input path.")
    parser.add_argument("-o", "--out_path", default="drawn.jpg", type=str, help="drawn image output path.")
    parser.add_argument("-b", "--box", type=str, help="input bounding box.")
    parser.add_argument(
        "-r",
        "--rate",
        default=0.25,
        type=float,
        help="highlight patch tokens which have top rate% similarity to cls token.")
    parser.add_argument("-p", "--palette", default="mako", type=str, help="select the shown palette.")
    parser.add_argument("-a", "--alpha", default=0.3, type=float, help="set blend coefficient.")

    return parser.parse_args()


def main():
    args = parse_args()

    path = args.path
    out_path = args.out_path
    boxes = args.box
    if boxes.endswith(".txt"):
        with open(boxes, "r") as fp:
            lines = fp.readlines()
            boxes = [tuple([int(x) for x in line.strip().split(",")]) for line in lines]
    else:
        boxes = [tuple([int(x) for x in args.box.strip().split(",")])]

    rate = float(args.rate)
    palette = args.palette
    alpha = args.alpha
    palette_set = ["rocket", "mako"]

    assert palette in palette_set

    raw_img = Image.open(path)
    drawn_img = np.array(raw_img.copy())

    for idx, box in enumerate(boxes):
        patch_img = np.array(raw_img)[box[1]:box[3], box[0]:box[2]]

        if len(boxes) > 1:
            palette = random.choice(palette_set)

        with torch.no_grad():
            cls_token, patch_tokens = model.encode_spatial_image(Image.fromarray(patch_img.astype(np.uint8)))
            # reduce batch dimension
            assert cls_token.shape[0] == 1 and patch_tokens.shape[0] == 1
            cls_token, patch_tokens = cls_token[0], patch_tokens[0]
            # acquire token spatial similarity and top rate% mask
            color_score, mask = top_percen_sim_tokens_vis(
                patch_img, cls_token, patch_tokens, rate=rate, palette=palette)

        # expand score & mask
        exp_color_score = np.zeros_like(drawn_img)
        exp_mask = np.zeros_like(drawn_img[:, :, 0])
        exp_color_score[box[1]:box[3], box[0]:box[2]] = color_score
        exp_mask[box[1]:box[3], box[0]:box[2]] = mask
        drawn_img = local_blend(drawn_img, exp_color_score, exp_mask, alpha)

        if palette == "mako":
            # cyan color
            drawn_img = cv2.rectangle(drawn_img, box[:2], box[2:4], color=(64, 183, 173), thickness=2)
        elif palette == "rocket":
            # red color
            drawn_img = cv2.rectangle(drawn_img, box[:2], box[2:4], color=(243, 118, 81), thickness=2)

        # box_drawn_img = local_blend(patch_img, color_score, mask, alpha)
        # Image.fromarray(box_drawn_img).save(f"{idx}.jpg")

    Image.fromarray(drawn_img).save(out_path)


if __name__ == "__main__":
    main()
