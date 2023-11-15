import os
import sqlite3
import torch
from torchvision import transforms
import numpy as np
from easydict import EasyDict


def join_str_to_make_query(train_sub_list):
    sql = [f"MetaData.Sub == {sub}" for sub in train_sub_list]
    return " OR ".join(sql)


class BCIC2b(torch.utils.data.Dataset):
    """
    * 769: left
    * 770: right
    """

    def __init__(
        self,
        args: EasyDict,
        is_test: bool = False,
        mode: str = "cls",
        transform: transforms.Compose = None,
    ):
        con = sqlite3.connect(os.path.join(args.DB_PATH, "bci2b.db"))
        cur = con.cursor()

        if mode == "cls":
            subject_list = [
                i for i in range(args.num_subjects) if i != args.target_subject
            ]
            np.random.seed(args.SEED)
            train_sub_list = np.random.choice(
                subject_list, size=(args.train_subject_num), replace=False
            )

            if args.is_subject_independent:
                if is_test:
                    cur.execute(
                        f"SELECT * FROM MetaData WHERE MetaData.Sub == {args.target_subject}"
                    )
                else:
                    cur.execute(
                        f"SELECT * FROM MetaData WHERE {join_str_to_make_query(train_sub_list)}"
                    )
            else:
                if is_test:
                    cur.execute(
                        f"SELECT * FROM MetaData WHERE MetaData.Sub == {args.target_subject} AND MetaData.Test == 1"
                    )
                else:
                    cur.execute(
                        f"SELECT * FROM MetaData WHERE MetaData.Sub == {args.target_subject} AND MetaData.Test == 0"
                    )

            self.metadata = cur.fetchall()
        elif mode == "ssl":
            cur.execute(f"SELECT * FROM MetaData")
            self.metadata = cur.fetchall()
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

        data = raw["data"][:, np.newaxis, :]
        label = np.array(raw["label"], dtype=np.int64)

        if self.transform is not None:
            data = self.transform(data)

        return data, label
