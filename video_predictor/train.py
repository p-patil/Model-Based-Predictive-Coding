import argparse

import torch
import tqdm

import video_predictor.data
import video_predictor.model


NUM_STEPS = 10**5
BATCH_SIZE = 128
VAL_EVERY = 3 * 10**2
NUM_TEST_BATCHES = 128


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--data-path", type=str, default="/data/ppatil32/atari_data")

    return ap.parse_args()


def evaluate(model, dataset, num_batches):
    model.eval()
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
    model = video_predictor.model.get_video_predictor()
    model = model.cuda()

    train_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/train", batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = video_predictor.data.AtariDataset(
        f"{args.data_path}/test", batch_size=BATCH_SIZE, shuffle=False)

    model.train()
    pbar = tqdm.tqdm(total=NUM_STEPS)
    step = 0
    best_val_loss = float("inf")
    while step < NUM_STEPS:
        batch = train_dataset.next_batch()
        batch["past"]["state"] = torch.from_numpy(batch["past"]["state"]).cuda()
        batch["future"]["state"] = torch.from_numpy(batch["future"]["state"]).cuda()
        batch["past"]["action"] =  torch.tensor(batch["past"]["action"]).cuda()
        batch["future"]["reward"] = torch.tensor(batch["future"]["reward"]).cuda()

        x = batch["past"]["state"].repeat((1, 3, 1, 1, 1))  # 3 channels
        y = batch["future"]["state"].squeeze()
        model.step(x, y, action=batch["past"]["action"], reward=batch["future"]["reward"])

        step += 1
        pbar.update(1)

        if (step + 1) % VAL_EVERY == 0:
            print(f"Step {step + 1} | Evaluating... ", end="")
            loss = evaluate(model, test_dataset, num_batches=NUM_TEST_BATCHES).item()
            model.train()
            print(f"Test loss = {loss}")

            if loss <= best_val_loss:
                save_path = Path(args.save_dir) / f"checkpoint{step}_{loss}.pt"
                torch.save(save_path)
                print(f"Saved checkpoint to: {save_path}")
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
