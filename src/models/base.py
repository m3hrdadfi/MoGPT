from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning.utilities import rank_zero_warn
from transformers import AutoConfig
from transformers import AutoModel
from transformers import PreTrainedTokenizerBase


class LiteTransformer(pl.LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        downstream_model_type: Optional[AutoModel] = None,
        load_weights: bool = True,
        lr: float = 1e-5,
        num_labels: int = 1,
        aux_num_labels: int = 1,
        no_decay: Optional[List] = None,
        weight_decay: float = 0.0,
        num_warmup_steps: int = 2000,
        num_freezing_layers: Optional[int] = 0,
        alpha: float = 1.0,
        **model_data_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.load_weights = load_weights
        self.model_data_kwargs = model_data_kwargs
        self.downstream_model_type = downstream_model_type
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self._tokenizer = tokenizer  # necessary for hf_pipeline

        self.initialize_model(self.pretrained_model_name_or_path, num_labels, aux_num_labels)

        self.num_labels = num_labels
        self.aux_num_labels = aux_num_labels

        self.lr = lr
        self.no_decay = no_decay if no_decay and isinstance(no_decay, list) else []
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_freezing_layers = num_freezing_layers

        self.alpha = alpha

    @property
    def tokenizer(self) -> Optional["PreTrainedTokenizerBase"]:
        if (
            self._tokenizer is None
            and hasattr(self, "trainer")  # noqa: W503
            and hasattr(self.trainer, "datamodule")  # noqa: W503
            and hasattr(self.trainer.datamodule, "tokenizer")  # noqa: W503
        ):
            self._tokenizer = self.trainer.datamodule.tokenizer
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: "PreTrainedTokenizerBase") -> None:
        self._tokenizer = tokenizer

    def resize_token_embeddings(self, model):
        tokenizer_length = len(self.tokenizer)
        model.resize_token_embeddings(tokenizer_length)

        return model

    def initialize_model(self, pretrained_model_name_or_path: str, num_labels: int, aux_num_labels: int):
        """create and initialize the model to use with this task,
        Feel free to overwrite this method if you are initializing the model in a different way
        """
        class_names = self.model_data_kwargs.get("class_names", [])
        if "class_names" in self.model_data_kwargs:
            del self.model_data_kwargs["class_names"]

        if self.load_weights:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
            setattr(config, "aux_num_labels", aux_num_labels)
            setattr(config, "class_names", class_names)

            model = self.downstream_model_type.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                **self.model_data_kwargs
            )
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels, **self.model_data_kwargs)
            setattr(config, "aux_num_labels", aux_num_labels)
            setattr(config, "class_name", class_names)

            model = self.downstream_model_type(config=config)

        self.model = self.resize_token_embeddings(model)

    def setup_no_decay(self, model, no_decay):
        """ Ignoring parameters from decay process"""
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        return parameters

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

    def configure_optimizers(self) -> Dict:
        rank_zero_warn(
            "You haven't specified an optimizer or lr scheduler. "
            "Defaulting to AdamW with an lr of 1e-5 and linear warmup for 10% of steps. "
            "To change this, override ``configure_optimizers`` in  TransformerModule."
        )
        parameters = self.setup_no_decay(self, self.no_decay)
        optimizer = torch.optim.AdamW(parameters, lr=self.lr)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=self.num_warmup_steps,
        )
        print(f"num_training_steps: {num_training_steps}, num_warmup_steps: {num_warmup_steps}")
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def configure_metrics(self, stage: str) -> Optional[Any]:
        """Override to configure metrics for train/validation/test.
        This is called on fit start to have access to the data module, and initialize any data specific metrics.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # tokenizer_length = len(self.tokenizer)
        # self.model.resize_token_embeddings(tokenizer_length)
        self.model.config.bos_token_id = self.tokenizer.vocab[self.tokenizer.bos_token]
        self.model.config.eos_token_id = self.tokenizer.vocab[self.tokenizer.eos_token]
        self.model.config.pad_token_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        self.configure_metrics(stage)

        if self.num_freezing_layers:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            for i, m in enumerate(self.model.transformer.h):
                if i >= self.num_freezing_layers:
                    for parameter in m.parameters():
                        parameter.requires_grad = True

            for parameter in self.model.transformer.ln_f.parameters():
                parameter.requires_grad = True

            for parameter in self.model.lm_head.parameters():
                parameter.requires_grad = True

            for parameter in self.model.classifier.parameters():
                parameter.requires_grad = True

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        model: LiteTransformer = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
        return model

    def save_hf_checkpoint(self, path: Union[str, Path]) -> None:
        """Save the model using the original HF AutoModel.
        This is useful for when you'd like to export the model to the hub.
        Args:
            path: Path to save the model to.
        """
        self.model.save_pretrained(path)
