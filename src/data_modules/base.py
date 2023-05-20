import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pytorch_lightning as pl
# from datasets import Dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class LoadDataset(Dataset):

    def __init__(
        self,
        json_path: str,
        convert_to_features: Callable,
        field: str = "data",
        n_samples: int = 0,
    ):
        self.data = LoadDataset.read_json_file(json_path, field=field, n_samples=n_samples)
        self.convert_to_features = convert_to_features

    @staticmethod
    def read_json_file(json_path, field=None, n_samples=0):
        if not os.path.exists(json_path):
            return []

        data = []
        with open(json_path, "r", encoding="utf-8") as fj:
            data = json.load(fj)
            if field:
                data = data.get(field, [])

        if isinstance(n_samples, int) and n_samples > 0:
            data = data[:n_samples]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        data = {k: [v] for k, v in data.items()}
        data = self.convert_to_features(data)
        data = {k: v[0] for k, v in data.items()}
        return data


class TransformerDataModule(pl.LightningDataModule):
    """Base ``LightningDataModule`` for HuggingFace Datasets. Provides helper functions and boilerplate logic to
        load/process datasets.
        Args:
            tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
        """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        ignore_index: int = -100,
        label_name: str = "labels",
        batch_size: int = 32,
        num_workers: int = 0,
        preprocessing_num_workers: int = 1,
        load_from_cache_file: bool = True,
        with_persona: bool = False,
        task_type: str = "NONE",
        cache_dir: Optional[Union[Path, str]] = None,
        limit_train_samples: Optional[int] = None,
        limit_val_samples: Optional[int] = None,
        limit_test_samples: Optional[int] = None,
        limit_predict_samples: Optional[int] = None,
        convert_to_features_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.predict_file = predict_file
        self.ignore_index = ignore_index
        self.label_name = label_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.cache_dir = cache_dir
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples
        self.limit_predict_samples = limit_predict_samples
        self.with_persona = with_persona
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"  # todo: smarter handling of this env variable

        self.convert_to_features_kwargs = convert_to_features_kwargs
        self.ds = {}
        self.task_type = task_type.upper()

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        # dataset = self._select_samples(dataset)
        # dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def load_dataset(self):
        data = {
            "train": LoadDataset(
                self.train_file,
                self.convert_to_features_for_train(
                    tokenizer=self.tokenizer,
                    with_persona=self.with_persona,
                    features_kwargs=self.convert_to_features_kwargs,
                ),
                field="data", n_samples=self.limit_train_samples
            ) if self.train_file else [],
            "validation": LoadDataset(
                self.validation_file,
                self.convert_to_features_for_validation(
                    tokenizer=self.tokenizer,
                    with_persona=self.with_persona,
                    features_kwargs=self.convert_to_features_kwargs,
                ),
                field="data", n_samples=self.limit_val_samples
            ) if self.validation_file else [],
            "test": LoadDataset(
                self.test_file,
                self.convert_to_features_for_test(
                    tokenizer=self.tokenizer,
                    with_persona=self.with_persona,
                    features_kwargs=self.convert_to_features_kwargs,
                ),
                field="data", n_samples=self.limit_test_samples
            ) if self.test_file else [],
            "predict": LoadDataset(
                self.test_file,
                self.convert_to_features_for_predict(
                    tokenizer=self.tokenizer,
                    with_persona=self.with_persona,
                    features_kwargs=self.convert_to_features_kwargs,
                ),
                field="data", n_samples=self.limit_predict_samples
            ) if self.test_file else [],
        }
        return data

    # def load_dataset(self) -> Dataset:
    #     # Allow custom data files when loading the dataset
    #     data_files, dataset_args, extension = {}, {}, None
    #     if self.train_file is not None:
    #         data_files["train"] = self.train_file
    #
    #         extension = self.train_file.split(".")[-1]
    #         if extension == "txt":
    #             extension = "text"
    #
    #         if extension == "json":
    #             dataset_args["field"] = "data"
    #
    #     if self.validation_file is not None:
    #         data_files["validation"] = self.validation_file
    #     if self.test_file is not None:
    #         data_files["test"] = self.test_file
    #
    #     if extension:
    #         dataset = load_dataset(extension, data_files=data_files, **dataset_args, cache_dir=self.cache_dir)
    #     else:
    #         raise MisconfigurationException(
    #             "You have not specified a dataset name nor a custom train and validation file"
    #         )
    #
    #     return dataset

    # def _select_samples(self, dataset: Dataset) -> Dataset:
    #     samples = (
    #         ("train", self.limit_train_samples),
    #         ("validation", self.limit_val_samples),
    #         ("test", self.limit_test_samples),
    #     )
    #     for column_name, n_samples in samples:
    #         if n_samples is not None and column_name in dataset:
    #             indices = range(min(len(dataset[column_name]), n_samples))
    #             dataset[column_name] = dataset[column_name].select(indices)
    #
    #     return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_for_train,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_for_validation,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn_for_test,
                shuffle=False,
            )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if "predict" in self.ds:
            return DataLoader(
                self.ds["predict"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn_for_predict,
                shuffle=False,
            )

    # def predict_dataloader(self) -> Optional[DataLoader]:
    #     if "predict" in self.ds:
    #         return DataLoader(
    #             self.ds["predict"],
    #             batch_size=self.batch_size,
    #             num_workers=self.num_workers,
    #             collate_fn=self.prediction_collate_fn,
    #         )

    @property
    def collate_fn_for_train(self) -> Optional[Callable]:
        raise NotImplementedError

    @property
    def collate_fn_for_validation(self) -> Optional[Callable]:
        raise NotImplementedError

    @property
    def collate_fn_for_test(self) -> Optional[Callable]:
        raise NotImplementedError

    @property
    def collate_fn_for_predict(self) -> Optional[Callable]:
        raise NotImplementedError

    # def process_data(self, dataset: Dataset, stage: Optional[str] = None) -> Dataset:
    #     column_names = dataset["train" if stage == "fit" else "validation"].column_names
    #
    #     if "train" in dataset:
    #         dataset["train"] = dataset["train"].map(
    #             self.convert_to_features_for_train(tokenizer=self.tokenizer, with_persona=self.with_persona, with_attention_mask=self.with_attention_mask,
    #                                                features_kwargs=self.convert_to_features_kwargs),
    #             batched=True,
    #             num_proc=self.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=self.load_from_cache_file,
    #         )
    #
    #     if "validation" in dataset:
    #         dataset["validation"] = dataset["validation"].map(
    #             self.convert_to_features_for_validation(tokenizer=self.tokenizer, with_persona=self.with_persona, with_attention_mask=self.with_attention_mask,
    #                                                     features_kwargs=self.convert_to_features_kwargs),
    #             batched=True,
    #             num_proc=self.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=self.load_from_cache_file,
    #         )
    #
    #     if "test" in dataset:
    #         dataset["test"] = dataset["test"].map(
    #             self.convert_to_features_for_test(tokenizer=self.tokenizer, with_persona=self.with_persona, features_kwargs=self.convert_to_features_kwargs),
    #             batched=True,
    #             num_proc=self.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=self.load_from_cache_file,
    #         )
    #
    #     return dataset

    @staticmethod
    def convert_to_features_for_train(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def convert_to_features_for_validation(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def convert_to_features_for_test(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def convert_to_features_for_predict(*args, **kwargs):
        raise NotImplementedError
