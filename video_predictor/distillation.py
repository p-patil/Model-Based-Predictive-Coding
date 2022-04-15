import argparse

import torch
import tqdm

import video_predictor.data
import video_predictor.model


NUM_STEPS = 10**5
BATCH_SIZE = 128
VAL_EVERY = 3 * 10**2
NUM_TEST_BATCHES = 128
DISTILLATION_PROB = 0.9

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--data-path", type=str, default="/data/ppatil32/atari_data")

    return ap.parse_args()


def evaluate(model, dataset, num_batches):
    loss = 0
    for _ in range(num_batches):
        batch = dataset.next_batch()
        batch["past"]["state"] = batch["past"]["state"].cuda()
        batch["future"]["state"] = batch["future"]["state"].cuda()
        batch["past"]["action"] =  batch["past"]["action"].cuda()
        batch["future"]["reward"] = batch["future"]["reward"].cuda()

        pred_frame, pred_reward = model(
            batch["past"]["state"], action=batch["past"]["action"])
        loss = torch.square(pred_frame - batch["future"]["state"]).mean().item()
        loss += torch.square(pred_reward - batch["future"]["reward"]).mean().item()
    loss /= len(num_batches)
    return loss


def main(args):
    teacher_model = video_predictor.model.get_video_predictor(pretrain=True).cuda()
    model = video_predictor.model.get_video_predictor(pretrain=True, small=True).cuda()

    train_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/train", batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/test", batch_size=BATCH_SIZE, shuffle=False)

    teacher_model.eval()
    model.train()
    pbar = tqdm.tqdm(total=NUM_STEPS)
    step = 0
    best_val_loss = float("inf")
    while step < NUM_STEPS:
        batch = train_dataset.next_batch()
        batch["past"]["state"] = torch.from_numpy(batch["past"]["state"]).cuda()
        batch["past"]["action"] =  torch.tensor(batch["past"]["action"]).cuda()

        x = batch["past"]["state"].repeat((1, 3, 1, 1, 1))  # 3 channels

        pred_frame, pred_reward = model.forward(x, action=batch["past"]["action"])
        if random.random() <= DISTILLATION_PROB:
            target_frame, target_reward = teacher_model.forward(x, action=batch["past"]["action"])
        else:
            target_frame, target_reward = batch["future"]["state"], batch["future"]["reward"]
        loss = (torch.square(pred_frame - target_frame).mean() +
                torch.square(pred_reward - target_reward).mean())
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        step += 1
        pbar.update(1)
        pbar.set_description(f"Loss = {loss.item()}")

        if (step + 1) % VAL_EVERY == 0:
            print(f"Step {step + 1} | Evaluating... ", end="")
            model.eval()
            loss = evaluate(model, test_dataset, num_batches=NUM_TEST_BATCHES)
            model.train()
            print(f"Test loss = {loss}")

            if loss <= best_val_loss:
                save_path = Path(args.save_dir) / f"checkpoint{step}_{loss}.pt"
                torch.save(save_path)
                print(f"Saved checkpoint to: {save_path}")
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
