import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import tqdm

import video_predictor.data
import video_predictor.model


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--data-path", type=str, default="../atari_data")
    ap.add_argument("--distributed", type=bool, default=False)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--distillation", action="store_true", default=False)
    ap.add_argument("--gt-weight", type=float, default=0.0)

    ap.add_argument("--num-steps", type=int, default=10**5)
    ap.add_argument("--num-test-batches", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)

    return ap.parse_args()


def evaluate(model, dataset, num_batches, device):
    '''
    Evaluate model on dataset and return mean loss
    '''
    model.eval()
    loss = 0
    for _ in range(num_batches):
        batch = dataset.next_batch()
        for key in ("state", "next_state", "action", "reward"):
            batch[key] = batch[key].to(device)

        x = batch["state"].repeat((1, 3, 1, 1, 1))  # 3 channels
        y = batch["next_state"].squeeze()

        pred_frame, pred_reward = model(x, action=batch["action"])
        loss += torch.square(pred_frame - y).mean().item()
        loss += torch.square(pred_reward - batch["reward"]).mean().item()
    return loss / num_batches


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.distillation:
        print(f"Running distillation with ground truth weight: {args.gt_weight}")
        assert args.checkpoint is not None
        teacher = video_predictor.model.get_video_predictor(chkpt=args.checkpoint)
        teacher.eval()
        model = video_predictor.model.get_video_predictor(small=True)
    else:
        model = video_predictor.model.get_video_predictor(chkpt=args.checkpoint)

    # Distributed training. TODO
    if args.distributed:
        raise NotImplementedError()
        print(f"{torch.cuda.device_count()} cuda devices available")
        model = nn.DataParallel(
            model,
            # device_ids=[
        )

    model = model.to(args.device)
    if args.distillation:
        teacher = teacher.to(args.device)

    train_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/train", batch_size=args.batch_size, shuffle=True, limit=args.limit)
    test_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/test", batch_size=args.batch_size, shuffle=False, limit=args.limit)

    model.train()
    pbar = tqdm.tqdm(total=args.num_steps, dynamic_ncols=True)
    step = 0
    best_val_loss = float("inf")
    while step < args.num_steps:
        batch = train_dataset.next_batch()
        for key in ("state", "next_state", "action", "reward"):
            batch[key] = batch[key].to(args.device)

        # States have shape (batch_size, num_channels, num_frames, height, width).
        x = batch["state"].repeat((1, 3, 1, 1, 1))  # 3 channels
        y = batch["next_state"].squeeze()

        if args.distillation:
            if args.distributed:
                raise NotImplementedError()
            else:
                pred_frame, pred_reward = model.forward(x, action=batch["action"])
                target_frame, target_reward = teacher.forward(x, action=batch["action"])
                loss = (1 - args.gt_weight) * (torch.square(pred_frame - target_frame).mean() +
                                               torch.square(pred_reward - target_reward).mean())
                loss += args.gt_weight * (torch.square(pred_frame - batch["next_state"]).mean() +
                                          torch.square(pred_reward - batch["reward"]).mean())
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

                loss = loss.item()
        else:
            if args.distributed:
                loss = model.module.step(x, y, action=batch["action"], reward=batch["reward"])
            else:
                loss = model.step(x, y, action=batch["action"], reward=batch["reward"])

        step += 1
        pbar.update(1)
        pbar.set_description(f"Train loss = {round(loss, 4)}")

        if (step + 1) % args.val_every == 0:
            print(f"Step {step + 1} | Evaluating... ", end="")
            loss = evaluate(model, test_dataset, num_batches=args.num_test_batches,
                            device=args.device)
            model.train()
            print(f"Test loss = {loss}")

            if loss <= best_val_loss:
                best_val_loss = loss
                save_path = save_dir / f"checkpoint{step}_{round(loss, 8)}.pt"
                with open(save_path, "wb") as f:
                    torch.save(model.state_dict(), f)
                print(f"New best validation loss, saved checkpoint to: {save_path}")
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
