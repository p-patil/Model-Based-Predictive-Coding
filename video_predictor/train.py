import argparse

import torch
import tqdm

import video_predictor.data
import video_predictor.model


DATA_PATH = "/data/ppatil32/atari_data"
NUM_WORKERS = 4

NUM_STEPS = 10**5
BATCH_SIZE = 128
VAL_EVERY = 3 * 10**2


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--save-dir", type=str, required=True)

    return ap.parse_args()


def evaluate(model, dataloader):
    model.eval()
    loss = 0
    for batch in dataloader:
        batch["past"]["state"] = batch["past"]["state"].cuda()
        batch["future"]["state"] = batch["future"]["state"].cuda()
        batch["past"]["action"] =  batch["past"]["action"].cuda()
        batch["future"]["reward"] = batch["future"]["reward"].cuda()

        pred_frame, pred_reward = model(
            batch["past"]["state"], action=batch["past"]["action"])
        loss = torch.square(pred_frame - batch["future"]["state"]).mean().item()
        loss += torch.square(pred_reward - batch["future"]["reward"]).mean().item()
    loss /= len(test_dataloader)
    return loss


def main(args):
    model = video_predictor.model.get_video_predictor()
    model = model.cuda()

    train_dataset = video_predictor.data.AtariDataset(f"{DATA_PATH}/train")
    test_dataset = video_predictor.data.AtariDataset(f"{DATA_PATH}/test")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        multiprocessing_context="spawn")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 8, shuffle=False, num_workers=NUM_WORKERS,
        multiprocessing_context="spawn")

    model.train()
    pbar = tqdm.tqdm(total=NUM_STEPS)
    step = 0
    best_val_loss = float("inf")
    while step < NUM_STEPS:
        for batch in train_dataloader:
            batch["past"]["state"] = batch["past"]["state"].cuda()
            batch["future"]["state"] = batch["future"]["state"].cuda()
            batch["past"]["action"] =  batch["past"]["action"].cuda()
            batch["future"]["reward"] = batch["future"]["reward"].cuda()

            batch["past"]["state"] = batch["past"]["state"].repeat((1, 3, 1, 1, 1))  # 3 channels
            batch["future"]["state"].squeeze_()
            model.step(batch["past"]["state"], batch["future"]["state"],
                       action=batch["past"]["action"], reward=batch["future"]["reward"])

            step += 1
            pbar.update(1)

            if (step + 1) % VAL_EVERY == 0:
                print(f"Step {step + 1} | Evaluating... ", end="")
                loss = evaluate(model, test_dataloader).item()
                model.train()
                print(f"Test loss = {loss}")

                if loss <= best_val_loss:
                    save_path = Path(args.save_dir) / f"checkpoint{step}_{loss}.pt"
                    torch.save(save_path)
                    print(f"Saved checkpoint to: {save_path}")
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
