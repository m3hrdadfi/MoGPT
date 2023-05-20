import copy
from itertools import chain
from itertools import repeat
from typing import Callable, Optional
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data_modules.base import TransformerDataModule
from data_modules.vanilla_gpt2 import DataCollatorWithPadding


def utterance_reordering(utterance_list: List, reorder_prob=0.10, do_auxiliary=False, offset=0, ignore_index=-100):
    if not len(utterance_list) > 1:
        return utterance_list, [[0]]

    utterance_list = copy.deepcopy(utterance_list)
    utterance_list_labels = list(map(lambda item: [0] * len(item), copy.deepcopy(utterance_list)))

    n = list(map(lambda item: item[0], filter(lambda item: len(item[1]) > 1, enumerate(utterance_list))))
    k = round(len(n) * reorder_prob)
    k = k if k > 0 else 1
    k = k - 1 if k >= len(n) else k
    indices = np.random.choice(n, k, replace=False)
    tokens_wrt_indices = {idx: utterance_list[idx] for idx in indices}
    reordered_tokens_wrt_indices = copy.deepcopy(tokens_wrt_indices)

    if do_auxiliary:
        reordered_tokens_wrt_indices = {idx: np.random.choice(tokens, len(tokens), replace=False).tolist() for idx, tokens in reordered_tokens_wrt_indices.items()}

    for idx in indices:
        utterance_list[idx] = reordered_tokens_wrt_indices[idx]
        utterance_list_labels[idx] = [1] * len(tokens_wrt_indices[idx])

    return utterance_list, utterance_list_labels


class DataCollatorForUtteranceReorderingWithPadding:
    def __init__(
        self,
        do_on,
        do_prob,
        reorder_prob,
        padding_index=0,
        ignore_index=-100,
        ignore_keys=None,
        zero_keys=None
    ):
        self.do_on = do_on
        self.do_prob = do_prob
        self.reorder_prob = reorder_prob
        self.padding_index = padding_index
        self.ignore_index = ignore_index
        self.ignore_keys = ignore_keys
        self.zero_keys = zero_keys

    def __call__(self, examples: List[Dict[str, List[int]]]):

        batch = {}
        for example in examples:
            if "input_ids" in example and "input_ids" not in batch:
                batch["input_ids"] = []
            if "labels" in example and "labels" not in batch:
                batch["labels"] = []
            if "length" in example and "length" not in batch:
                batch["length"] = []
            if "aux_labels" in example and "aux_labels" not in batch:
                batch["aux_labels"] = []

            input_ids, labels, aux_labels, length, persona_sep, context_sep, len_persona, len_context = (
                example["input_ids"],
                example["labels"],
                example["aux_labels"],
                example["length"],
                example["persona_sep"],
                example["context_sep"],
                example["len_persona"],
                example["len_context"],
            )
            before_persona = input_ids[:persona_sep[0]]
            persona_part = input_ids[persona_sep[0]: persona_sep[1]]
            context_part = input_ids[context_sep[0]: context_sep[1]]
            after_context = input_ids[context_sep[1]:]

            aug_persona_part, labels_persona_part = persona_part, list(map(lambda item: [0] * len(item), copy.deepcopy(persona_part)))
            aug_context_part, labels_context_part = context_part, list(map(lambda item: [0] * len(item), copy.deepcopy(context_part)))

            do_prob = np.random.rand()
            if do_prob < self.do_prob:
                do_auxiliary = True
            else:
                do_auxiliary = False

            if self.do_on == "CONTEXT":

                if len_context > 1:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=context_part[0][0],
                    )
            elif self.do_on == "PERSONA":
                if len_persona > 1:
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=persona_part[0][0],
                    )
            elif self.do_on == "PERSONA+CONTEXT" or self.do_on == "CONTEXT+PERSONA":
                if len_persona > 1 and len_context > 1:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                elif len_persona > 1 and not len_context > 1:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=False,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                elif not len_persona > 1 and len_context > 1:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=False,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=False,
                        ignore_index=context_part[0][0],
                    )
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=False,
                        ignore_index=persona_part[0][0],
                    )
            else:
                prob = np.random.rand()
                if prob < 0.5:
                    aug_context_part, labels_context_part = utterance_reordering(
                        utterance_list=context_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary if len_context > 1 else False,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_persona_part, labels_persona_part = utterance_reordering(
                        utterance_list=persona_part,
                        reorder_prob=self.reorder_prob,
                        do_auxiliary=do_auxiliary if len_persona > 1 else False,
                        ignore_index=self.ignore_index,
                    )

            aux_labels = (
                [0] * len(list(chain(*before_persona))) +
                list(chain(*labels_persona_part)) +
                list(chain(*labels_context_part)) +
                [0] * len(list(chain(*after_context)))
            )

            # input_ids = list(chain(*input_ids))
            input_ids = list(chain(*(before_persona + aug_persona_part + aug_context_part + after_context)))

            batch["input_ids"].append(torch.LongTensor(input_ids))
            batch["labels"].append(torch.LongTensor(labels))
            batch["aux_labels"].append(torch.LongTensor(aux_labels))
            batch["length"].append(torch.LongTensor(length))

        for key in batch:
            if key not in self.ignore_keys and key not in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.padding_index)
            elif key in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            else:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.ignore_index)

        return batch


class UtteranceReorderingDataModule(TransformerDataModule):

    def __init__(
        self,
        convert_to_features_kwargs,
        do_on,
        do_prob,
        reorder_prob,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.convert_to_features_kwargs = convert_to_features_kwargs
        self.do_on = do_on
        self.do_prob = do_prob
        self.reorder_prob = reorder_prob

    @property
    def collate_fn_for_train(self) -> Optional[Callable]:
        return DataCollatorForUtteranceReorderingWithPadding(
            do_on=self.do_on,
            do_prob=self.do_prob,
            reorder_prob=self.reorder_prob,
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
            data = {"input_ids": [], "labels": [], "aux_labels": [], "length": [], "len_persona": [], "len_context": [], "persona_sep": [], "context_sep": []}

            for persona, candidates, history in zip(examples[persona_col_name], examples[candidates_col_name], examples[history_col_name]):
                persona = [persona_token] + persona if with_persona else [persona_token]

                context, u1, u2 = history[: -1], history[-1:], candidates[-1:]
                context = [dialog_token] + context
                dialog = [dialog_token] + u1 + u2

                persona_length = 1 + (len(persona) - 1) * 2 - 1 if len(persona) > 1 else 1
                context_length = 1 + (len(context) - 1) * 2 - 1 if len(context) > 1 else 1

                persona_sep = [1, 1 + persona_length]
                context_sep = [1 + persona_length, 1 + persona_length + context_length]

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

                data["input_ids"].append(input_ids)
                data["labels"].append(labels)
                data["persona_sep"].append(persona_sep)
                data["context_sep"].append(context_sep)
                data["len_persona"].append(len(persona))
                data["len_context"].append(len(context))
                data["aux_labels"].append([])
                data["length"].append([len(input_ids)])

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
