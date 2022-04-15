import os
import os.path as osp

import mmcv
import torch
import torch.nn as nn

import sys
sys.path.insert(0, "Video-Swin-Transformer/")
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import build_optimizer


# TODO(piyush) This model's output feature map is 3x3 which seems really small.
CONFIG_FILE = "~/Model-Based-Predictive-Coding/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py"
WORK_DIR = "work_dir"
PRETRAIN_PATH = "Video-Swin-Transformer/pretrain/swin_base_patch244_window1677_kinetics400_22k.pth"


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

        return loss


def get_video_predictor(pretrain=True, small=False):
    if small:
        raise NotImplementedError() # TODO(piyush) Implement small model

    cfg = Config.fromfile(CONFIG_FILE)

    cfg.work_dir = WORK_DIR

    # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))


    if pretrain:
        # cfg.model.backbone.pretrained = PRETRAIN_PATH
        cfg.load_from = PRETRAIN_PATH
        cfg.model.backbone.use_checkpoint = True

    model = VideoPredictor(cfg)

    return model


# imgs = torch.rand(32, 3, 4, 224, 224)
# y = torch.rand(32, 3, 224, 224)
# model = get_video_predictor()

# imgs = imgs.to("cuda")
# y = y.to("cuda")
# model = model.to("cuda")

# for i in range(1000):
    # loss = model.step(imgs, y)
    # if i % 25 == 0:
        # print(i, loss.item())
