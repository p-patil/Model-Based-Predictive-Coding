import os
import os.path as osp
import re

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "Video-Swin-Transformer/")
import mmcv
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import build_optimizer, load_state_dict


WORK_DIR = "work_dir"
CONFIG_FILE = "~/Model-Based-Predictive-Coding/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py"
PRETRAIN_PATH = "pretrain/swin_base_patch244_window1677_kinetics400_22k.pth"
SMALL_CONFIG_FILE = "~/Model-Based-Predictive-Coding/Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
SMALL_PRETRAIN_PATH = "pretrain/swin_tiny_patch244_window877_kinetics400_1k.pth"


class VideoPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.swin_dim = 1024 * 2

        self.featurizer = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        self.decoder = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(in_channels=self.swin_dim, out_channels=self.swin_dim // 4,
                               kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=self.swin_dim // 4),
            nn.LeakyReLU(),
            # Block 2
            nn.ConvTranspose2d(in_channels=self.swin_dim // 4, out_channels=self.swin_dim // 4**2,
                               kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=self.swin_dim // 4**2),
            nn.LeakyReLU(),
            # Block 3
            nn.ConvTranspose2d(in_channels=self.swin_dim // 4**2, out_channels=self.swin_dim // 4**3,
                               kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=self.swin_dim // 4**3),
            nn.LeakyReLU(),
            # Block 4 (next 4 frames + reward = 5 output channels)
            nn.ConvTranspose2d(in_channels=self.swin_dim // 4**3,
                               out_channels=5, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(num_features=5),
            nn.LeakyReLU(),
        )

        # TODO(piyush) Use our own optimizer?
        self.optimizer = build_optimizer(self, cfg.optimizer)

    def forward(self, x, action=None):
        x = self.featurizer.extract_feat(x)  # (32, 1024, 2, 7, 7)
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        if action is not None:
            pass  # TODO(piyush)
        x = self.decoder(x)

        # Crop 84x84 image from center.
        if x.shape[-1] > 84:
            d = (x.shape[-1] - 84) // 2
            x = x[..., d : -d - 1, d : -d - 1]

        pred_frame, pred_reward = x[:, :-1, ...], x[:, -1, ...]
        pred_reward = pred_reward.reshape(*pred_reward.shape[:-2], -1).mean(dim=-1)  # GAP
        return pred_frame, pred_reward

    def step(self, x, next_frame, action=None, reward=None):
        pred_frame, pred_reward = self.forward(x, action=action)
        loss = torch.square(pred_frame - next_frame).mean()
        if reward is not None:
            loss = loss + torch.square(pred_reward - reward).mean()

        # TODO(piyush) LR scheduling?
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def get_video_predictor(chkpt=None, distributed=False, pretrain=True, small=False):
    if small:
        cfg = Config.fromfile(SMALL_CONFIG_FILE)
    else:
        cfg = Config.fromfile(CONFIG_FILE)

    cfg.work_dir = WORK_DIR

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Create model.
    model = VideoPredictor(cfg)

    if chkpt is not None:
        print(f"Loading weights from checkpoint: {chkpt}")
        model.load_state_dict(torch.load(chkpt), strict=False)
    elif pretrain:
        print("Using pretrained weights")
        if small:
            checkpoint = torch.load(SMALL_PRETRAIN_PATH)
        else:
            checkpoint = torch.load(PRETRAIN_PATH)

        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {filename}')

        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip prefix of state_dict.
        revise_keys = [(r'^module\.', '')]
        for p, r in revise_keys:
            state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

        load_state_dict(model.featurizer, state_dict)
    else:
        print("Failed to use pretrain state dict")

    if distributed:
        model = nn.DataParallel(model, gpu_ids = [3, 4])
    return model
