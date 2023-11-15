import os
import sqlite3
import torch
from torchvision import transforms
import numpy as np
from easydict import EasyDict


class SleepEDF(torch.utils.data.Dataset):
    def __init__(
        self,
        args: EasyDict,
        is_test: bool = False,
        mode: str = "cls",
        transform: transforms.Compose = None,
    ):
        if mode == "cls":
            if is_test:
                con = sqlite3.connect(
                    os.path.join(
                        args.DB_PATH,
                        f"sleep_edf_test_{args.inter_information_length}.db",
                    )
                )
                cur = con.cursor()

                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub == {args.target_subject}"
                )
            else:
                con = sqlite3.connect(
                    os.path.join(
                        args.DB_PATH,
                        f"sleep_edf_train_{args.inter_information_length}.db",
                    )
                )
                cur = con.cursor()

                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub != {args.target_subject}"
                )

            self.metadata = cur.fetchall()
        elif mode == "ssl":
            con = sqlite3.connect(
                os.path.join(
                    args.DB_PATH, f"sleep_edf_test_{args.inter_information_length}.db"
                )
            )
            cur = con.cursor()
            cur.execute(f"SELECT * FROM MetaData")
            test_data = cur.fetchall()

            con = sqlite3.connect(
                os.path.join(
                    args.DB_PATH, f"sleep_edf_train_{args.inter_information_length}.db"
                )
            )
            cur = con.cursor()
            cur.execute(f"SELECT * FROM MetaData")
            train_data = cur.fetchall()

            self.metadata = train_data + test_data

        else:
            raise ValueError("mode should be either 'cls' or 'ssl'")

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
