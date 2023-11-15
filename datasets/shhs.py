import os
import sqlite3
import torch
from torchvision import transforms
import numpy as np
from easydict import EasyDict


class SHHS(torch.utils.data.Dataset):
    def __init__(
        self,
        args: EasyDict,
        is_test: bool = False,
        mode: str = "cls",
        transform: transforms.Compose = None,
    ):
        con = sqlite3.connect(
            os.path.join(
                args.DB_PATH,
                f"shhs.db",
            )
        )
        cur = con.cursor()
        if mode == "cls":
            np.random.seed(42)
            random_idx = np.arange(5793)
            np.random.shuffle(random_idx)
            train_idx = list(map(str, random_idx[:4634]))
            test_idx = list(map(str, random_idx[4634:]))
            train_idx_str = ",".join(train_idx)
            test_idx_str = ",".join(test_idx)

            if is_test:
                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub IN ({test_idx_str})"
                )
            else:
                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub IN ({train_idx_str})"
                )
        elif mode == "ssl":
            cur.execute("SELECT * FROM MetaData")
        else:
            raise ValueError("mode should be either 'cls' or 'ssl'")

        self.metadata = cur.fetchall()

        print("LOG >>> Successfully connected to the database")

        self.transform = transform

        cur.close()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.metadata[idx][-1]
        raw = np.load(filename)

        data = raw["data"]
        label = np.array(raw["label"].item().split(","), dtype=np.int64)

        if self.transform is not None:
            data = self.transform(data)

        return data, label
