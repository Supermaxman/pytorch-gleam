import base64
import hashlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import Type

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        tokenizer_name: str = None,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        predict_path: str = None,
        batch_size: int = 32,
        max_seq_len: int = 512,
        num_workers: int = 8,
        use_tpus: bool = False,
    ):
        super().__init__()

        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.use_tpus = use_tpus
        self.tokenizer = None
        if self.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    @staticmethod
    def load_or_create(cls: Type, data_path: str, **kwargs):
        key_kwargs = {k: v for k, v in kwargs.items() if k != "pickle_path"}
        if "pickle_path" not in kwargs or kwargs["pickle_path"] is None:
            return cls(**key_kwargs)

        pickle_path = kwargs["pickle_path"]
        class_repr = f"{cls}" + "|".join(
            [f"{k}-({str(v)})" for k, v in key_kwargs.items()]
        )
        ex_hasher = hashlib.sha1(class_repr.encode("utf-8"))
        ex_hash = ex_hasher.digest()[:12]
        file_name = base64.urlsafe_b64encode(ex_hash).decode("utf-8") + ".pickle"
        file_path = os.path.join(pickle_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            return obj
        else:
            obj = cls(**key_kwargs)
            obj.load(data_path)
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
            return obj

    @abstractmethod
    def create_collator(self):
        pass

    def get_datasets(self, ds):
        if not isinstance(ds, list):
            ds = [ds]
        return ds

    def flatten_dataloaders(self, data_loaders):
        if isinstance(data_loaders, list):
            if len(data_loaders) == 1:
                data_loaders = data_loaders[0]
        return data_loaders

    def create_eval_data_loaders(self, datasets):
        data_loaders = [
            DataLoader(
                ds,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.create_collator(),
                worker_init_fn=ds.worker_init_fn,
                # ensures same samples because rng will get assigned during worker creation
                persistent_workers=False,
                pin_memory=True,
            )
            for ds in datasets
        ]
        return data_loaders

    def create_train_data_loaders(self, datasets):
        data_loaders = [
            DataLoader(
                ds,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=self.create_collator(),
                worker_init_fn=ds.worker_init_fn,
                # ensures different samples across epochs from rng generator
                # seeded on creation with worker seed
                persistent_workers=True,
                pin_memory=True,
            )
            for ds in datasets
        ]
        return data_loaders

    def train_dataloader(self):
        data_sets = self.get_datasets(self.train_dataset)
        data_loaders = self.create_train_data_loaders(data_sets)
        data_loaders = self.flatten_dataloaders(data_loaders)
        return data_loaders

    def val_dataloader(self):
        data_sets = self.get_datasets(self.val_dataset)
        data_loaders = self.create_eval_data_loaders(data_sets)
        data_loaders = self.flatten_dataloaders(data_loaders)
        return data_loaders

    def test_dataloader(self):
        data_sets = self.get_datasets(self.test_dataset)
        data_loaders = self.create_eval_data_loaders(data_sets)
        data_loaders = self.flatten_dataloaders(data_loaders)
        return data_loaders

    def predict_dataloader(self):
        data_sets = self.get_datasets(self.predict_dataset)
        data_loaders = self.create_eval_data_loaders(data_sets)
        data_loaders = self.flatten_dataloaders(data_loaders)
        return data_loaders
