"""Import libraries"""
import os
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning import seed_everything, Trainer
import datasets.eeg_transforms as e_transforms
from models.litmodel import LitModelAutoEncoder
from models.init import get_model
from utils.setup_utils import get_device
from utils.training_utils import get_configs
from const import dataset_info_dict


""" Argparse """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bci2a')
aargs = parser.parse_args()

""" Config setting """
CONFIG_PATH = f"{os.getcwd()}/configs"
filename = dataset_info_dict[aargs.dataset]['config']

""" Change """
model_info = dataset_info_dict[aargs.dataset]['model']
args = get_configs(config_path=CONFIG_PATH, filename=filename, dataset=model_info)
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


def load_data(dataset: Dataset):
    transform = transforms.Compose(
        [
            e_transforms.MinMaxNormalization(),
            e_transforms.ChannelPermutation(),
            e_transforms.ToTensor(),
        ]
    )

    train_dataset = dataset(
        args=args,
        is_test=False,
        mode="ssl",
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader


def main():
    """ Change """
    dataset = dataset_info_dict[aargs.dataset]['dataset']
    train_dataloader = load_data(dataset=dataset)
    sample_data = next(iter(train_dataloader))

    ### Load Model ###
    encoder = get_model(args=args)

    model = LitModelAutoEncoder(model=encoder, sample_data=sample_data[0], args=args)

    ### Training ###
    devices = list(map(int, args.GPU_NUM.split(",")))
    trainer = Trainer(
        max_epochs=args.EPOCHS,
        accelerator="gpu",
        devices=devices,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
    )


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
