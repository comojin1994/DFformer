import os
import sqlite3
import traceback
from tqdm import tqdm
from glob import glob
import numpy as np

BASE_PATH = "/opt/pytorch/Datasets/SHHS"
SAVE_PATH = "/opt/pytorch/Datasets/SHHS_preprocessed"
DB_PATH = "/opt/pytorch/DFformer/databases/shhs.db"

EPOCH_LENGTH = 20

con = sqlite3.connect(DB_PATH)
print("LOG >>> Successfully connected to the training database")

cur = con.cursor()
print("LOG >>> Successfully created Table")

cur.execute(
    """CREATE TABLE MetaData(
    Sub Integer,
    Label text,
    Path text);"""
)

filelist = sorted(glob(BASE_PATH + "/*.npz"))
file_idx = 0

pbar = tqdm(filelist)
for subject_id, filename in enumerate(pbar):
    try:
        pbar.set_postfix({"Filename": filename})
        if filename == "/opt/pytorch/Datasets/SHHS/shhs1-203135.npz":
            continue
        raw = np.load(filename)

        x, y = raw["x"], raw["y"]

        for jdx in range(0, x.shape[0], EPOCH_LENGTH):
            if jdx + EPOCH_LENGTH >= x.shape[0]:
                current_data = x[jdx:]
                insufficient_num = EPOCH_LENGTH - current_data.shape[0]
                padding_data = np.repeat(
                    current_data[-1:], repeats=insufficient_num, axis=0
                )
                current_data = np.concatenate((current_data, padding_data), axis=0)

                current_label = y[jdx:]
                padding_label = np.repeat(9, repeats=insufficient_num, axis=0)
                current_label = np.concatenate((current_label, padding_label), axis=0)

            else:
                current_data = x[jdx : jdx + EPOCH_LENGTH]
                current_label = y[jdx : jdx + EPOCH_LENGTH]

            current_data = current_data.transpose(0, 2, 1)
            current_label = list(map(str, current_label.tolist()))
            current_label = ",".join(current_label)

            save_filename = os.path.join(SAVE_PATH, f"{file_idx:06d}.npz")
            np.savez(save_filename, data=current_data, label=current_label)

            cur.execute(
                "INSERT INTO MetaData Values(:Sub, :Label, :Path)",
                {
                    "Sub": subject_id,
                    "Label": current_label,
                    "Path": save_filename,
                },
            )

            file_idx += 1

            if current_data.min() == current_data.max():
                import pdb

                pdb.set_trace()
            if current_data.min() - current_data.max() == 0:
                import pdb

                pdb.set_trace()

    except Exception as e:
        print("LOG >>> Error occured while processing: ", filename)
        print(e)
        print(traceback.format_exc())
        continue

con.commit()
con.close()
