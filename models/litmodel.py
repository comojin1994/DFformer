import math
import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from easydict import EasyDict
import matplotlib.pyplot as plt
import mlflow


class LitModelLinear(LightningModule):
    def __init__(self, model: nn.Module, args: EasyDict):
        super().__init__()

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, x):
        return self.model(x)

    def cal_logit_label(self, logit, label):
        logit = logit[:, 0, :]
        logit = logit.reshape(-1, self.args.num_classes)

        return logit, label

    def training_step(self, batch, batch_idx):
        self._adjust_learning_rate()

        label = batch[1].reshape(-1)

        logit = self(batch[0])

        logit, label = self.cal_logit_label(logit, label)

        pred = logit.argmax(axis=1)

        loss = self.criterion(logit, label)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()
        acc = accuracy(pred, label)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            self.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def evaluate(self, batch, stage=None):
        label = batch[1].reshape(-1)

        logit = self(batch[0])
        # if logit.dim() > 2:
        logit, label = self.cal_logit_label(logit, label)

        pred = logit.argmax(axis=1)

        loss = self.criterion(logit, label)
        acc = accuracy(pred, label)

        if stage:
            self.log(
                f"{stage}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage="eval")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logit = self(batch[0])
        logit, _ = self.cal_logit_label(logit, None)
        pred = logit.argmax(axis=1)

        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return {"optimizer": self.optimizer}

    def _adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        warmup_epoch = self.args.EPOCHS // 10 if self.args.EPOCHS <= 100 else 40

        if self.current_epoch < warmup_epoch:
            cur_lr = self.args.lr * self.current_epoch / warmup_epoch + 1e-9
        else:
            cur_lr = (
                self.args.lr
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (self.current_epoch - warmup_epoch)
                        / (self.args.EPOCHS - warmup_epoch)
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr


class LitSleepLinear(LitModelLinear):
    def cal_logit_label(self, logit, label):
        if logit.dim() > 2:
            logit = logit[:, 2:, :]
            logit = logit.reshape(-1, self.args.num_classes)
        if label is not None and 9 in label:
            non_label_9 = label != 9
            label = label[non_label_9]
            logit = logit[non_label_9]

        # SHHS 예외처리
        if label is not None and 6 in label:
            non_label_6 = label != 6
            label = label[non_label_6]
            logit = logit[non_label_6]

        return logit, label


class LitMILinear(LitModelLinear):
    def cal_logit_label(self, logit, label):
        if logit.dim() > 2:
            logit = logit[:, 0, :]
            logit = logit.reshape(-1, self.args.num_classes)

        return logit, label
    

class LitModelAutoEncoder(LightningModule):
    def __init__(self, model, sample_data, args):
        super().__init__()

        self.model = model
        self.criterion = nn.L1Loss()
        self.sample_data = sample_data
        self.args = args

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.adjust_learning_rate()
        pred = self(batch[0])
        loss = self.criterion(pred, batch[0])

        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        if self.current_epoch % 10 == 0:
            sample_pred = self(
                self.sample_data.type_as(
                    self.state_dict()["model.encoder.embedding.temporal_pos_embed"]
                )
            )

            fig, axs = plt.subplots(
                self.args.inter_information_length, figsize=(30, 10)
            )

            for i in range(self.args.inter_information_length):
                axs[i].plot(self.sample_data[4, i, 0])
                axs[i].plot(sample_pred[4, i, 0].detach().cpu().numpy())
                if i < self.args.inter_information_length - 1:
                    plt.setp(axs[i].get_xticklabels(), visible=False)
                    # Remove the x-axis line on the first subplot
                    axs[i].spines["bottom"].set_visible(False)
                if i > 0:
                    # Remove the upper line on the second subplot
                    axs[i].spines["top"].set_visible(False)
                plt.setp(axs[i].get_yticklabels(), visible=False)

            # Remove the space between the subplots
            plt.subplots_adjust(hspace=0)

            fig.canvas.draw()
            figure = np.array(fig.canvas.renderer._renderer)

            self.logger.experiment.add_image(
                "Tracker", np.transpose(figure, (2, 0, 1)), self.current_epoch
            )
            mlflow.log_image(figure, f"sample_data/{self.current_epoch:02d}.png")

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.95),
        )

        return {"optimizer": self.optimizer}

    def adjust_learning_rate(self):
        cur_lr = (
            self.args.lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.current_epoch / self.args.EPOCHS))
        )

        for param_group in self.optimizer.param_groups:
            if "fix_lr" in param_group and param_group["fix_lr"]:
                param_group["lr"] = self.args.lr
            else:
                param_group["lr"] = cur_lr