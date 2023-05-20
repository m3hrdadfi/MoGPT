import json
import os
import sys
from argparse import Namespace
from typing import Optional

import pytorch_lightning as pl
from huggingface_hub import Repository
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from transformers import PreTrainedTokenizerBase


class GenerateResponseAndPushModelToHubCallback(Callback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_loader,
        wandb_logger: Optional[WandbLogger] = None,
        bar_length=30,
        args: Optional[Namespace] = None,
        repo: Optional[Repository] = None,
        **generation_kwargs
    ):
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.wandb_logger = wandb_logger
        self.bar_length = bar_length
        self.repo = repo
        self.args = args
        self.generation_kwargs = generation_kwargs

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        data_table = []
        counter = 1
        total = len(self.data_loader)

        for idx, batch in enumerate(self.data_loader):
            idx = idx + 1
            percent = 100.0 * idx / total
            sys.stdout.write("\r")
            sys.stdout.write("Completed: [{:{}}] {:>3}%".format('=' * int(percent / (100.0 / self.bar_length)), self.bar_length, int(percent)))

            for i in range(batch["input_ids"].shape[0]):
                batch_input = {
                    k: v[i][v[i] != self.tokenizer.pad_token_id].unsqueeze(dim=0).to(pl_module.device)
                    if k == "input_ids" else
                    v[i].unsqueeze(dim=0).to(pl_module.device)
                    for k, v in batch.items()
                }
                response = batch_input["labels"].squeeze(dim=0).to(pl_module.device)
                del batch_input["labels"]

                g_responses = pl_module.model.generate(**batch_input, **self.generation_kwargs)
                for j, g_response in enumerate(g_responses):
                    data_table.append([
                        f"epoch_{trainer.current_epoch + 1}-item_{counter}-gen_{j + 1}",
                        self.tokenizer.decode(batch_input["input_ids"][0]).replace(self.tokenizer.pad_token, "").strip(),
                        self.tokenizer.decode(response[response > -1]),
                        self.tokenizer.decode(g_response[len(batch_input["input_ids"][0]):]).replace(self.tokenizer.pad_token, "").strip()
                    ])

                counter += 1

            sys.stdout.flush()

        output_json = os.path.join(self.args.output_dir, "outputs")
        os.makedirs(output_json, exist_ok=True)
        with open(os.path.join(output_json, f"epoch_{trainer.current_epoch + 1}.json"), "w") as fj:
            json.dump({"output": data_table}, fj)

        if self.wandb_logger:
            self.wandb_logger.log_table(
                key=f"epoch_{trainer.current_epoch + 1}",
                columns=["state", "dialog", "r_response", "g_response"],
                data=data_table
            )

        if self.args:
            # pass
            pl_module.save_hf_checkpoint(self.args.output_dir)
            pl_module.tokenizer.save_pretrained(self.args.output_dir)
