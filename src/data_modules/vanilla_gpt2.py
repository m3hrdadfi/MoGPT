from itertools import chain
from itertools import repeat
from typing import Callable, Optional
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from data_modules.base import TransformerDataModule


class DataCollatorWithPadding:
    def __init__(
        self,
        padding_index=0,
        ignore_index=-100,
        ignore_keys=None,
        zero_keys=None,
    ):
        self.padding_index = padding_index
        self.ignore_index = ignore_index
        self.ignore_keys = ignore_keys
        self.zero_keys = zero_keys

    def __call__(self, examples: List[Dict[str, List[int]]]):
        batch = {}
        for example in examples:
            for key in example:
                if key not in batch:
                    batch[key] = []

                batch[key].append(torch.LongTensor(example[key]))

        for key in batch:
            if key not in self.ignore_keys and key not in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.padding_index)
            elif key in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            else:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.ignore_index)

        return batch


class VanillaGPT2DataModule(TransformerDataModule):

    def __init__(
        self,
        convert_to_features_kwargs,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.convert_to_features_kwargs = convert_to_features_kwargs

    @property
    def collate_fn_for_train(self) -> Optional[Callable]:
        return DataCollatorWithPadding(
            padding_index=self.tokenizer.pad_token_id,
            ignore_index=self.convert_to_features_kwargs["ignore_index"],
            ignore_keys=self.convert_to_features_kwargs["ignore_keys"],
            zero_keys=self.convert_to_features_kwargs["zero_keys"],
        )

    @property
    def collate_fn_for_validation(self) -> Optional[Callable]:
        return DataCollatorWithPadding(
            padding_index=self.tokenizer.pad_token_id,
            ignore_index=self.convert_to_features_kwargs["ignore_index"],
            ignore_keys=self.convert_to_features_kwargs["ignore_keys"],
            zero_keys=self.convert_to_features_kwargs["zero_keys"],
        )

    @property
    def collate_fn_for_test(self) -> Optional[Callable]:
        return DataCollatorWithPadding(
            padding_index=self.tokenizer.pad_token_id,
            ignore_index=self.convert_to_features_kwargs["ignore_index"],
            ignore_keys=self.convert_to_features_kwargs["ignore_keys"],
            zero_keys=self.convert_to_features_kwargs["zero_keys"],
        )

    @property
    def collate_fn_for_predict(self) -> Optional[Callable]:
        return DataCollatorWithPadding(
            padding_index=self.tokenizer.pad_token_id,
            ignore_index=self.convert_to_features_kwargs["ignore_index"],
            ignore_keys=self.convert_to_features_kwargs["ignore_keys"],
            zero_keys=self.convert_to_features_kwargs["zero_keys"],
        )

    @staticmethod
    def convert_to_features_for_train(
        tokenizer: Dict,
        with_persona: bool = False,
        persona_col_name: str = "personality",
        candidates_col_name: str = "candidates",
        history_col_name: str = "history",
        features_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        features_kwargs = features_kwargs if features_kwargs else {}
        persona_token = features_kwargs["persona_token"]
        context_token = features_kwargs["context_token"]
        dialog_token = features_kwargs["dialog_token"]

        bos_token = features_kwargs["bos_token"]
        eos_token = features_kwargs["eos_token"]
        sep_token = features_kwargs["sep_token"]

        ignore_index = features_kwargs["ignore_index"]

        def fn(examples):
            data = {"input_ids": [], "labels": []}

            for persona, candidates, history in zip(examples[persona_col_name], examples[candidates_col_name], examples[history_col_name]):
                persona = [persona_token] + persona if with_persona else [persona_token]

                context, u1, u2 = history[:-1], history[-1:], candidates[-1:]
                # context = [context_token] + context
                context = [dialog_token] + context
                dialog = [dialog_token] + u1 + u2

                input_sequence = (
                    [bos_token] +
                    persona[:1] +
                    list(chain(*zip(persona[1:], repeat(sep_token))))[:-1] +
                    context[:1] +
                    (list(chain(*zip(context[1:], repeat(sep_token)))) if len(context) > 1 else list(chain(*zip(context[1:], repeat(sep_token))))[:-1]) +
                    # dialog[:1] +
                    list(chain(*zip(dialog[1:], repeat(sep_token))))[:-1] +
                    [eos_token]
                )

                input_ids = tokenizer(input_sequence).input_ids
                labels = [ignore_index] * len(list(chain(*input_ids[:-2]))) + input_ids[-2] + input_ids[-1]
                input_ids = list(chain(*input_ids))

                data["input_ids"].append(input_ids)
                data["labels"].append(labels)

            return data

        return fn

    @staticmethod
    def convert_to_features_for_validation(
        tokenizer: Dict,
        with_persona: bool = False,
        persona_col_name: str = "personality",
        candidates_col_name: str = "candidates",
        history_col_name: str = "history",
        features_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        features_kwargs = features_kwargs if features_kwargs else {}
        persona_token = features_kwargs["persona_token"]
        context_token = features_kwargs["context_token"]
        dialog_token = features_kwargs["dialog_token"]

        bos_token = features_kwargs["bos_token"]
        eos_token = features_kwargs["eos_token"]
        sep_token = features_kwargs["sep_token"]

        ignore_index = features_kwargs["ignore_index"]

        def fn(examples):
            data = {"input_ids": [], "labels": []}

            for persona, candidates, history in zip(examples[persona_col_name], examples[candidates_col_name], examples[history_col_name]):
                persona = [persona_token] + persona if with_persona else [persona_token]

                context, u1, u2 = history[:-1], history[-1:], candidates[-1:]
                # context = [context_token] + context
                context = [dialog_token] + context
                dialog = [dialog_token] + u1 + u2

                input_sequence = (
                    [bos_token] +
                    persona[:1] +
                    list(chain(*zip(persona[1:], repeat(sep_token))))[:-1] +
                    context[:1] +
                    (list(chain(*zip(context[1:], repeat(sep_token)))) if len(context) > 1 else list(chain(*zip(context[1:], repeat(sep_token))))[:-1]) +
                    # dialog[:1] +
                    list(chain(*zip(dialog[1:], repeat(sep_token))))[:-1] +
                    [eos_token]
                )

                input_ids = tokenizer(input_sequence).input_ids
                labels = [ignore_index] * len(list(chain(*input_ids[:-2]))) + input_ids[-2] + input_ids[-1]
                input_ids = list(chain(*input_ids))

                data["input_ids"].append(input_ids)
                data["labels"].append(labels)

            return data

        return fn

    @staticmethod
    def convert_to_features_for_test(
        tokenizer: Dict,
        with_persona: bool = False,
        persona_col_name: str = "personality",
        candidates_col_name: str = "candidates",
        history_col_name: str = "history",
        features_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        features_kwargs = features_kwargs if features_kwargs else {}
        persona_token = features_kwargs["persona_token"]
        context_token = features_kwargs["context_token"]
        dialog_token = features_kwargs["dialog_token"]

        bos_token = features_kwargs["bos_token"]
        eos_token = features_kwargs["eos_token"]
        sep_token = features_kwargs["sep_token"]

        ignore_index = features_kwargs["ignore_index"]

        def fn(examples):
            data = {"input_ids": [], "labels": []}

            for persona, candidates, history in zip(examples[persona_col_name], examples[candidates_col_name], examples[history_col_name]):
                persona = [persona_token] + persona if with_persona else [persona_token]

                context, u1, u2 = history[:-1], history[-1:], candidates[-1:]
                # context = [context_token] + context
                context = [dialog_token] + context
                dialog = [dialog_token] + u1 + u2

                input_sequence = (
                    [bos_token] +
                    persona[:1] +
                    list(chain(*zip(persona[1:], repeat(sep_token))))[:-1] +
                    context[:1] +
                    (list(chain(*zip(context[1:], repeat(sep_token)))) if len(context) > 1 else list(chain(*zip(context[1:], repeat(sep_token))))[:-1]) +
                    # dialog[:1] +
                    list(chain(*zip(dialog[1:], repeat(sep_token))))[:-1] +
                    [eos_token]
                )

                input_ids = tokenizer(input_sequence).input_ids
                labels = [ignore_index] * len(list(chain(*input_ids[:-2]))) + input_ids[-2] + input_ids[-1]
                input_ids = list(chain(*input_ids))

                data["input_ids"].append(input_ids)
                data["labels"].append(labels)

            return data

        return fn

    @staticmethod
    def convert_to_features_for_predict(
        tokenizer: Dict,
        with_persona: bool = False,
        persona_col_name: str = "personality",
        candidates_col_name: str = "candidates",
        history_col_name: str = "history",
        features_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        features_kwargs = features_kwargs if features_kwargs else {}
        persona_token = features_kwargs["persona_token"]
        context_token = features_kwargs["context_token"]
        dialog_token = features_kwargs["dialog_token"]

        bos_token = features_kwargs["bos_token"]
        eos_token = features_kwargs["eos_token"]
        sep_token = features_kwargs["sep_token"]

        ignore_index = features_kwargs["ignore_index"]

        def fn(examples):
            data = {"input_ids": [], "labels": []}

            for persona, candidates, history in zip(examples[persona_col_name], examples[candidates_col_name], examples[history_col_name]):
                persona = [persona_token] + persona if with_persona else [persona_token]

                context, u1, u2 = history[:-1], history[-1:], candidates[-1:]
                # context = [context_token] + context
                context = [dialog_token] + context
                dialog = [dialog_token] + u1 + u2

                input_sequence = (
                    [bos_token] +
                    persona[:1] +
                    list(chain(*zip(persona[1:], repeat(sep_token))))[:-1] +
                    context[:1] +
                    (list(chain(*zip(context[1:], repeat(sep_token)))) if len(context) > 1 else list(chain(*zip(context[1:], repeat(sep_token))))[:-1]) +
                    # dialog[:1] +
                    list(chain(*zip(dialog[1:], repeat(sep_token))))[:-1] +
                    [eos_token]
                )

                input_ids = tokenizer(input_sequence).input_ids
                labels = input_ids[-2]
                input_ids = list(chain(*input_ids[:-2]))

                data["input_ids"].append(input_ids)
                data["labels"].append(labels)

            return data

        return fn
