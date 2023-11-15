import os
import sqlite3
import traceback
import mne
from tqdm import tqdm
from glob import glob
import numpy as np
from braindecode.datautil.preprocess import exponential_moving_standardize
from scipy.io import loadmat


BASE_PATH = "/opt/pytorch/Datasets/BCI_Competition_IV/BCI_Competition_IV_2b"
SAVE_PATH = (
    "/opt/pytorch/Datasets/BCI_Competition_IV/BCI_Competition_IV_2b_preprocessed"
)
DB_PATH = "/opt/pytorch/DFformer/databases/bci2b.db"

origin_ival = [0, 1000]
file_idx = 0

con = sqlite3.connect(DB_PATH)
print("LOG >>> Successfully connected to the database")

cur = con.cursor()
print("LOG >>> Successfully created Table")

cur.execute(
    """CREATE TABLE MetaData(
    Sub Integer,
    Test Integer,
    Path text
    );"""
)

filelist = sorted(glob(f"{BASE_PATH}/*.gdf"))
labellist = sorted(glob(f"{BASE_PATH}/true_labels/*.mat"))

pbar = tqdm(filelist)
for idx, filename in enumerate(pbar):
    pbar.set_postfix({"Filename": filename})
    try:
        # Read Raws
        raw = mne.io.read_raw_gdf(filename, preload=True, verbose=False, include="EEG")
        events, annot = mne.events_from_annotations(raw, verbose=False)

        # Bandpass filtering
        raw.filter(0.0, 38.0, fir_design="firwin", verbose=False)

        # Epoching
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
        )

        # Trigger setting
        tmin, tmax = (
            origin_ival[0] / raw.info["sfreq"],
            (origin_ival[1] - 1) / raw.info["sfreq"],
        )

        if "T.gdf" in filename:
            event_id = dict({"769": annot["769"], "770": annot["770"]})
        elif "E.gdf" in filename:
            event_id = dict({"783": annot["783"]})
        else:
            raise NotImplementedError("Invalid filename")

        # Get data with epochs format
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            baseline=None,
            picks=picks,
            preload=True,
            verbose=False,
        )

        # Scaling: 1e6
        epochs_data = epochs.get_data() * 1e6

        # Normalization: Exponential moving standardize
        preprocessed_data = []
        for epoch in epochs_data:
            epoch = exponential_moving_standardize(
                epoch, init_block_size=int(epochs.info["sfreq"] * 4)
            )
            preprocessed_data.append(epoch)
        preprocessed_data = np.stack(preprocessed_data)
        preprocessed_data = (preprocessed_data - preprocessed_data.min()) / (
            preprocessed_data.max() - preprocessed_data.min()
        )

        label_list = loadmat(labellist[idx])["classlabel"].reshape(-1) - 1

        for i in range(preprocessed_data.shape[0]):
            save_filename = os.path.join(SAVE_PATH, f"{file_idx:06d}.npz")
            np.savez(save_filename, data=preprocessed_data[i], label=label_list[i])

            cur.execute(
                "INSERT INTO MetaData Values(:Sub, :Test, :Path)",
                {
                    "Sub": int(filename.split("/")[-1][1:3]) - 1,
                    "Test": 1 if "E.gdf" in filename else 0,
                    "Path": save_filename,
                },
            )

            file_idx += 1
    except Exception as e:
        print(filename)
        print(e)
        print(traceback.format_exc())

con.commit()
con.close()
