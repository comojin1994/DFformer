"""Import libraries"""
import os
from datetime import datetime
from models.backbones.dfformer import calculate_output_size
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import seed_everything, Trainer
from datasets.sleepedf import SleepEDF
import datasets.eeg_transforms as e_transforms
from models.litmodel import LitSleepLinear
from models.init import get_model
from utils.setup_utils import get_device
from utils.training_utils import get_configs

""" Config setting """
CONFIG_PATH = f"{os.getcwd()}/configs"
filename = "sleepedf_config.yaml"

args = get_configs(config_path=CONFIG_PATH, filename=filename, dataset="SleepEDF")
args.current_time = datetime.now().strftime("%Y%m%d")

# Set Device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NUM

args["device"] = get_device(args.GPU_NUM)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deteministic = True

args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)

# Set SEED
seed_everything(args.SEED)


def load_data(num_subject: int):
    args.target_subject = num_subject

    transform = transforms.Compose(
        [e_transforms.ToTensor(), e_transforms.MinMaxNormalization()]
    )

    dataset = SleepEDF(args=args, is_test=False, transform=transform)
    seed_everything(args.SEED)
    train_data_size = int(len(dataset) * args.train_data_ratio / 100)
    train_dataset, _ = random_split(
        dataset, [train_data_size, len(dataset) - train_data_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = SleepEDF(args=args, is_test=True, transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, test_dataloader


def load_ckpt(model: nn.Module, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        if k.startswith("model.encoder."):
            state_dict[k[len("model.encoder.") :]] = state_dict[k]
        del state_dict[k]

    for k in list(state_dict.keys()):
        if k.startswith("classifier_head."):
            del state_dict[k]

    ### Reshape pos embedding ###
    if state_dict["embedding.temporal_pos_embed"].shape[1] < args.seq_len + 1:
        seq_len = calculate_output_size(args.seq_len, args.cnn_layers)
        state_dict["embedding.temporal_pos_embed"] = F.interpolate(
            state_dict["embedding.temporal_pos_embed"][np.newaxis, ...],
            size=(seq_len + 1, args.dim),
        )[0]
    else:
        state_dict["embedding.temporal_pos_embed"] = state_dict[
            "embedding.temporal_pos_embed"
        ][:, : args.seq_len + 1, :]
    if (
        state_dict["embedding.spatial_pos_embed"].shape[1]
        < args.inter_information_length
    ):
        state_dict["embedding.spatial_pos_embed"] = F.interpolate(
            state_dict["embedding.spatial_pos_embed"][np.newaxis, ...],
            size=(args.inter_information_length + 1, args.dim),
        )[0]
    else:
        state_dict["embedding.spatial_pos_embed"] = state_dict[
            "embedding.spatial_pos_embed"
        ][:, : args.inter_information_length + 1, :]

    msg = model.load_state_dict(state_dict, strict=False)

    print(f"LOG >>>\n{msg}")

    return model


def main():
    total_results = []

    for num_subject in range(args.num_subjects):
        train_dataloader, test_dataloader = load_data(num_subject=num_subject)

        ### Load Model ###
        encoder = get_model(
            args=args, load_ckpt=load_ckpt if args.WEIGHT_PATH is not None else None
        )
        model = LitSleepLinear(model=encoder, args=args)

        ### Training ###
        devices = list(map(int, args.GPU_NUM.split(",")))
        trainer = Trainer(
            max_epochs=args.EPOCHS,
            accelerator="gpu",
            devices=devices,
            default_root_dir=f"{args.CKPT_PATH}/sleepedf",
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader,
        )

        # Evaluate in test data
        result = trainer.test(model, dataloaders=test_dataloader)
        acc = result[0]["eval_acc"]
        total_results.append(acc)

    total_results = list(map(lambda x: f"{x:.4f}", total_results))
    total_results = ",".join(total_results)
    print(f"Total results: {total_results}")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
