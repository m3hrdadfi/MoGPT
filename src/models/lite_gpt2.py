import math

import torch
from transformers import AutoModel
from transformers import GPT2LMHeadModel

from models.base import LiteTransformer


class LiteGPT2LMHeadModel(LiteTransformer):

    def __init__(
        self,
        downstream_model_type: AutoModel = GPT2LMHeadModel,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            downstream_model_type=downstream_model_type,
            *args,
            **kwargs
        )
        self.save_hyperparameters()

    def _step(self, batch):
        outputs = self.model(**batch)
        lm_loss = outputs.loss
        return lm_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("train_loss_step", loss)
        self.log("train_lm_loss_step", loss)
        self.log("train_perplexity_step", perplexity)

        output_dict = {"loss": loss, "lm_loss": loss}
        return output_dict

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("train_loss_epoch", loss)
        self.log("train_lm_loss_epoch", lm_loss)
        self.log("train_perplexity_epoch", perplexity)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch)

        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("val_loss_step", loss)
        self.log("val_lm_loss_step", loss)
        self.log("val_perplexity_step", perplexity)

        output_dict = {"loss": loss, "lm_loss": loss}
        return output_dict

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("val_loss_epoch", loss)
        self.log("val_lm_loss_epoch", lm_loss)
        self.log("val_perplexity_epoch", perplexity)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._step(batch)

        self.log("test_loss_step", loss)
        self.log("test_lm_loss_step", loss)

        output_dict = {"loss": loss, "lm_loss": loss}
        return output_dict

    def test_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("test_loss_epoch", loss)
        self.log("test_lm_loss_epoch", lm_loss)
        self.log("test_perplexity_epoch", perplexity)
