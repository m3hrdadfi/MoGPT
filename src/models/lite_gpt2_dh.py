import math

import torch
from transformers import AutoModel

from models.base import LiteTransformer
from models.gpt2_dh import GPT2DoubleHeadsModel


class LiteGPT2DoubleHeadsModel(LiteTransformer):
    def __init__(
        self,
        *args,
        downstream_model_type: AutoModel = GPT2DoubleHeadsModel,
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
        aux_loss = outputs.aux_loss if outputs.aux_loss else torch.tensor(0.0).to(self.device)
        loss = lm_loss + self.alpha * aux_loss

        return loss, lm_loss, aux_loss

    def training_step(self, batch, batch_idx):
        loss, lm_loss, aux_loss = self._step(batch)

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("train_loss_step", loss)
        self.log("train_lm_loss_step", lm_loss)
        self.log("train_aux_loss_step", aux_loss)
        self.log("train_perplexity_step", perplexity)

        output_dict = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}
        return output_dict

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))
        aux_loss = torch.mean(torch.stack([output["aux_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("train_loss_epoch", loss)
        self.log("train_lm_loss_epoch", lm_loss)
        self.log("train_aux_loss_epoch", aux_loss)
        self.log("train_perplexity_epoch", perplexity)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, lm_loss, aux_loss = self._step(batch)

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("val_loss_step", loss)
        self.log("val_lm_loss_step", lm_loss)
        self.log("val_aux_loss_step", aux_loss)
        self.log("val_perplexity_step", perplexity)

        output_dict = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}
        return output_dict

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))
        aux_loss = torch.mean(torch.stack([output["aux_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("val_loss_epoch", loss)
        self.log("val_lm_loss_epoch", lm_loss)
        self.log("val_aux_loss_epoch", aux_loss)
        self.log("val_perplexity_epoch", perplexity)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, lm_loss, aux_loss = self._step(batch)

        self.log("test_loss_step", loss)
        self.log("tet_lm_loss_step", lm_loss)
        self.log("test_aux_loss_step", aux_loss)

        output_dict = {"loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}
        return output_dict

    def test_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        lm_loss = torch.mean(torch.stack([output["lm_loss"] for output in outputs]))
        aux_loss = torch.mean(torch.stack([output["aux_loss"] for output in outputs]))

        try:
            perplexity = math.exp(lm_loss)
        except OverflowError:
            perplexity = float("inf")

        self.log("test_loss_epoch", loss)
        self.log("test_lm_loss_epoch", lm_loss)
        self.log("test_aux_loss_epoch", aux_loss)
        self.log("test_perplexity_epoch", perplexity)
