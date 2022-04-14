import pickle
from pathlib import Path

import torch


DATA_PER_FILE = 1000


class AtariDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        assert self.data_path.is_dir()

        self.data_files = list(self.data_path.glob("*"))

    def __getitem__(self, idx):
        file_idx, data_idx = idx // DATA_PER_FILE, idx % DATA_PER_FILE
        with open(self.data_files[file_idx], "rb") as f:
            data = pickle.load(f)

        if data_idx == DATA_PER_FILE - 1:
            data_idx -= 1
        past, future = data[data_idx], data[data_idx + 1]

        return {"past": past, "future": future}

    def __len__(self):
        return len(self.data_files) * DATA_PER_FILE
