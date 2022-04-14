import pickle
from pathlib import Path

import torch


DATA_PER_FILE = 1000


class AtariDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, out_channels=None):
        self.data_path = Path(data_path)
        self.out_channels = out_channels
        assert self.data_path.is_dir()

        self.data_files = list(self.data_path.glob("*"))

    # TODO(piyush) Implement caching?
    def __getitem__(self, idx):
        file_idx, data_idx = idx // DATA_PER_FILE, idx % DATA_PER_FILE
        with open(self.data_files[file_idx], "rb") as f:
            data = pickle.load(f)

        if data_idx == DATA_PER_FILE - 1:
            data_idx -= 1
        past, future = data[data_idx], data[data_idx + 1]

        if self.out_channels is not None:
            past["state"] = past["state"].repeat((self.out_channels, 1, 1, 1))
            future["state"] = future["state"].repeat((self.out_channels, 1, 1, 1))

        return {"past": past, "future": future}

    def __len__(self):
        return len(self.data_files) * DATA_PER_FILE
