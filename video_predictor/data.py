import concurrent.futures
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import tqdm


BLOCK_LEN = 1000


class AtariDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, batch_size=32, shuffle=False, limit=None):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        assert self.data_path.is_dir()

        self.data_files = list(self.data_path.glob("*"))
        if shuffle:
            random.shuffle(self.data_files)
        if limit is not None:
            print(f"Only using {limit} data blocks")
            self.data_files = self.data_files[: limit]

        self.cache_len = batch_size
        self.cache_index = 0
        self.block_index = 0
        self.cache = []
        self.reload_cache(self.cache_len)

    def reload_cache(self, cache_size, multithread=False):
        def read_block_file(filename):
            with open(filename, "rb") as f:
                block = pickle.load(f)
            assert len(block) == BLOCK_LEN
            return block

        print("Reloading cache... ", end="")
        start = time.time()
        if multithread:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(read_block_file, self.data_files[i])
                    for i in range(self.cache_index, self.cache_index + self.batch_size)
                ]
            self.cache = [future.result() for future in tqdm.tqdm(futures)]
        else:
            self.cache = [
                read_block_file(filepath)
                for filepath in tqdm.tqdm(self.data_files[self.cache_index :
                                                          self.cache_index + self.cache_len])
            ]
        end = time.time()
        print(f"Done ({end - start} sec)")

        self.cache_index += self.batch_size
        if self.cache_index >= len(self.data_files):
            print("Finished a full pass over dataset")
            self.cache_index = 0
            self.reload_cache(self.cache_len)

    def next_batch(self):
        samples = [block[self.block_index] for block in self.cache]
        batch =  {
            key: torch.from_numpy(np.array([sample[key] for sample in samples]))
            for key in samples[0].keys()
        }

        self.block_index += 1
        if self.block_index == BLOCK_LEN:
            self.reload_cache(self.cache_len)
            self.block_index = 0

        return batch

    def __len__(self):
        return len(self.data_files) * BLOCK_LEN
