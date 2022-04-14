import torch
import tqdm

import video_predictor.data
import video_predictor.model


DATA_PATH = "/data/ppatil32/atari_data"
NUM_WORKERS = 0

NUM_STEPS = 10**5
BATCH_SIZE = 32
VAL_EVERY = 10**3


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


def main():
    model = video_predictor.model.get_video_predictor()
    model = model.cuda()

    train_dataset = video_predictor.data.AtariDataset(f"{DATA_PATH}/train", out_channels=3)
    test_dataset = video_predictor.data.AtariDataset(f"{DATA_PATH}/test", out_channels=3)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 8, shuffle=False, num_workers=NUM_WORKERS)

    model.train()
    pbar = tqdm.tqdm(total=NUM_STEPS)
    step = 0
    while step < NUM_STEPS:
        for batch in train_dataloader:
            batch["past"]["state"] = batch["past"]["state"].cuda()
            batch["future"]["state"] = batch["future"]["state"].cuda()
            batch["past"]["action"] =  batch["past"]["action"].cuda()
            batch["future"]["reward"] = batch["future"]["reward"].cuda()

            model.step(batch["past"]["state"], batch["future"]["state"],
                       action=batch["past"]["action"], reward=batch["future"]["reward"])

            step += 1
            pbar.update(1)

            if (step + 1) % VAL_EVERY == 0:
                print(f"Step {step + 1} | Evaluating... ", end="")
                loss = evaluate(model, test_dataloader)
                model.train()
                print(f"Test loss = {loss.item()}")
    pbar.close()



if __name__ == "__main__":
    main()
