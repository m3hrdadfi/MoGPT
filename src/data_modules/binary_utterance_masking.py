import copy
import numpy as np
import torch
from data_modules.base import TransformerDataModule
from data_modules.vanilla_gpt2 import DataCollatorWithPadding
from itertools import chain
from itertools import repeat
from nltk.corpus import wordnet
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline
from typing import Callable, Optional
from typing import Dict, List

filler = pipeline("fill-mask")


def word_synonym_antonym(token):
    is_cap = token[0].isupper()
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets(token):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    synonyms = list(map(lambda item: item.lower().capitalize().replace("_", " ") if is_cap else item.lower().replace("_", " "), list(dict.fromkeys(synonyms))))
    synonyms = list(filter(lambda item: item != token, synonyms))

    antonyms = list(map(lambda item: item.lower().capitalize().replace("_", " ") if is_cap else item.lower().replace("_", " "), list(dict.fromkeys(antonyms))))
    antonyms = list(filter(lambda item: item != token, antonyms))

    return synonyms, antonyms


def utterance_masking(tokenizer, utterance_list: List, do_prob=0.10, masked_prob=0.8, changed_prob=0.5, ignore_index=-100):
    if not len(utterance_list) > 1:
        utterance_list = tokenizer(utterance_list).input_ids
        utterance_list_labels = [[0]]
        return utterance_list, utterance_list_labels

    utterance_list = list(map(lambda item: item.split(), copy.deepcopy(utterance_list)))

    only_utterances = list(map(lambda item: item[0], filter(lambda item: len(item[1]) > 1, enumerate(utterance_list))))
    only_utterances_with_position = []
    for i in only_utterances:
        for j, token in enumerate(utterance_list[i]):
            only_utterances_with_position.append((i, j, token))

    rand = np.random.rand(len(only_utterances_with_position))
    masking_indices = [only_utterances_with_position[i] for i, m in enumerate(rand < do_prob) if m]

    masked_list = []
    for mask in masking_indices:
        i, j, token = mask
        token_id, changed_token_id, label = tokenizer(token).input_ids, tokenizer(token).input_ids, [0]
        if np.random.rand() < masked_prob:
            candidates = word_synonym_antonym(token)
            if len(candidates[0]) > 0:
                changed_token_id = tokenizer(candidates[0][0]).input_ids
                label = [1] * len(changed_token_id)
            else:
                changed_token_id = token_id
                label = [0] * len(token_id)
        else:
            if np.random.rand() < changed_prob:
                candidates = word_synonym_antonym(token)
                if len(candidates[-1]) > 0:
                    changed_token_id = tokenizer(candidates[-1][0]).input_ids
                    label = [2] * len(changed_token_id)
                else:
                    changed_token_id = token_id
                    label = [0] * len(token_id)
            else:
                changed_token_id = token_id
                label = [0] * len(token_id)
        masked_list.append((i, j, label, token_id, changed_token_id))

    utterance_list = list(map(lambda item: tokenizer(item).input_ids if isinstance(item, list) else tokenizer([item]).input_ids, utterance_list))
    # utterance_list_labels = list(map(lambda item: [(np.array(x) * 0).tolist() for x in item], utterance_list))
    utterance_list_labels = list(map(lambda item: [[0 for x in sub_item] for sub_item in item], utterance_list))

    for mask in masked_list:
        i, j, label, token, changed_token = mask
        utterance_list[i][j] = changed_token
        utterance_list_labels[i][j] = label

    utterance_list = [list(chain(*sub_item)) for sub_item in utterance_list]
    utterance_list_labels = [list(chain(*sub_item)) for sub_item in utterance_list_labels]

    return utterance_list, utterance_list_labels


class DataCollatorForUtteranceMaskingWithPadding:
    def __init__(
        self,
        tokenizer,
        do_on,
        do_prob,
        masked_prob,
        changed_prob,
        vocab,
        mask_token_id,
        padding_index=0,
        ignore_index=-100,
        ignore_keys=None,
        zero_keys=None
    ):
        self.tokenizer = tokenizer
        self.do_on = do_on
        self.do_prob = do_prob
        self.masked_prob = masked_prob
        self.changed_prob = changed_prob
        self.vocab = vocab
        self.mask_token_id = mask_token_id
        self.padding_index = padding_index
        self.ignore_index = ignore_index
        self.ignore_keys = ignore_keys
        self.zero_keys = zero_keys

    def __call__(self, examples: List[Dict[str, List[int]]]):

        batch = {"input_ids": [], "labels": [], "aux_labels": [], "length": []}
        for example in examples:
            input_sequence, persona_sep, context_sep, len_persona, len_context = (
                example["input_sequence"],
                example["persona_sep"],
                example["context_sep"],
                example["len_persona"],
                example["len_context"],
            )

            before_persona = input_sequence[:persona_sep[0]]
            persona_part = input_sequence[persona_sep[0]: persona_sep[1]]
            context_part = input_sequence[context_sep[0]: context_sep[1]]
            after_context = input_sequence[context_sep[1]:]

            before_persona = [
                list(chain(*sub_item))
                for sub_item in list(map(lambda item: self.tokenizer(item).input_ids if isinstance(item, list) else self.tokenizer([item]).input_ids, before_persona))
            ]
            after_context = [
                list(chain(*sub_item))
                for sub_item in list(map(lambda item: self.tokenizer(item).input_ids if isinstance(item, list) else self.tokenizer([item]).input_ids, after_context))
            ]

            aug_persona_part = [
                list(chain(*sub_item))
                for sub_item in list(map(lambda item: self.tokenizer(item).input_ids if isinstance(item, list) else self.tokenizer([item]).input_ids, persona_part))
            ]
            aug_context_part = [
                list(chain(*sub_item))
                for sub_item in list(map(lambda item: self.tokenizer(item).input_ids if isinstance(item, list) else self.tokenizer([item]).input_ids, context_part))
            ]
            labels_persona_part = list(map(lambda item: [0] * len(item), aug_persona_part))
            labels_context_part = list(map(lambda item: [0] * len(item), aug_context_part))

            if self.do_on == "CONTEXT":

                if len_context > 1:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=context_part[0][0],
                    )
            elif self.do_on == "PERSONA":
                if len_persona > 1:
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=persona_part[0][0],
                    )
            elif self.do_on == "PERSONA+CONTEXT" or self.do_on == "CONTEXT+PERSONA":
                if len_persona > 1 and len_context > 1:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                elif len_persona > 1 and not len_context > 1:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                elif not len_persona > 1 and len_context > 1:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=context_part[0][0],
                    )
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=persona_part[0][0],
                    )
            else:
                prob = np.random.rand()
                if prob < 0.5:
                    aug_context_part, labels_context_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=context_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )
                else:
                    aug_persona_part, labels_persona_part = utterance_masking(
                        tokenizer=self.tokenizer,
                        utterance_list=persona_part,
                        do_prob=self.do_prob,
                        masked_prob=self.masked_prob,
                        changed_prob=self.changed_prob,
                        ignore_index=self.ignore_index,
                    )

            aux_labels = (
                [0] * len(list(chain(*before_persona))) +
                list(chain(*labels_persona_part)) +
                list(chain(*labels_context_part)) +
                [0] * len(list(chain(*after_context)))
            )

            # input_ids = list(chain(*input_ids))
            input_ids = before_persona + aug_persona_part + aug_context_part + after_context
            labels = [self.ignore_index] * len(list(chain(*input_ids[:-2]))) + input_ids[-2] + input_ids[-1]
            input_ids = list(chain(*input_ids))

            batch["input_ids"].append(torch.LongTensor(input_ids))
            batch["labels"].append(torch.LongTensor(labels))
            batch["aux_labels"].append(torch.LongTensor(aux_labels))
            batch["length"].append(torch.LongTensor(len(input_ids)))

        for key in batch:
            if key not in self.ignore_keys and key not in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.padding_index)
            elif key in self.zero_keys:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            else:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=self.ignore_index)

        return batch


class UtteranceMaskingDataModule(TransformerDataModule):

    def __init__(
        self,
        convert_to_features_kwargs,
        do_on,
        do_prob,
        masked_prob,
        changed_prob,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        special_tokens_ids = [self.tokenizer.vocab[sp] for sp in list(chain(*[[sp] if isinstance(sp, str) else sp for sp in list(self.tokenizer.special_tokens_map.values())]))]
        vocab = [i for i in range(len(self.tokenizer)) if i not in special_tokens_ids]
        self.convert_to_features_kwargs = convert_to_features_kwargs
        self.convert_to_features_kwargs["vocab"] = vocab
        self.do_on = do_on
        self.do_prob = do_prob
        self.masked_prob = masked_prob
        self.changed_prob = changed_prob
        self.vocab = vocab

    @property
    def collate_fn_for_train(self) -> Optional[Callable]:
        return DataCollatorForUtteranceMaskingWithPadding(
            tokenizer=self.tokenizer,
            do_on=self.do_on,
            do_prob=self.do_prob,
            masked_prob=self.masked_prob,
            changed_prob=self.changed_prob,
            vocab=self.vocab,
            mask_token_id=self.tokenizer.mask_token_id,
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
            # data = {"input_ids": [], "labels": [], "aux_labels": [], "length": [], "len_persona": [], "len_context": [], "persona_sep": [], "context_sep": []}
            data = {"input_sequence": [], "len_persona": [], "len_context": [], "persona_sep": [], "context_sep": []}

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

                # input_ids = tokenizer(input_sequence).input_ids
                # labels = [ignore_index] * len(list(chain(*input_ids[:-2]))) + input_ids[-2] + input_ids[-1]

                # data["input_ids"].append(input_ids)
                # data["labels"].append(labels)

                data["input_sequence"].append(input_sequence)
                data["persona_sep"].append(persona_sep)
                data["context_sep"].append(context_sep)
                data["len_persona"].append(len(persona))
                data["len_context"].append(len(context))

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
