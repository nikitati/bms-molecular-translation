from typing import List, Optional
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.io
from torchvision import transforms
import albumentations as A
import pytorch_lightning as pl
import cv2
from PIL import Image
import selfies
from tqdm import tqdm

PRETRAINED_MEAN = [0.485, 0.456, 0.406]
PRETRAINED_STD = [0.229, 0.224, 0.225]
PRETRAINED_TEST_MEAN = [0.485, 0.456, 0.406]
PRETRAINED_TEST_STD = [0.229, 0.224, 0.225]


def get_img_file_path(
        image_id: str,
        split: str,
        dataset_path: str = "../data/raw"
):
    return "{}/{}/{}/{}/{}/{}.png".format(
        dataset_path, split, image_id[0], image_id[1], image_id[2], image_id
    )


def load_targets(targets_path: str, target: str) -> pd.DataFrame:
    usecols = list({"image_id", "InChI", target})
    df_targets = pd.read_csv(
        targets_path,
        usecols=usecols
    )
    df_targets.rename(columns={target: "target"}, inplace=True)
    return df_targets


class Tokenizer:

    def __init__(self):
        self.stoi = dict()
        self.itos = dict()
        self.aux_tokens = ['<SOS>', '<EOS>', '<UNK>', '<PAD>']

    def __len__(self):
        return len(self.stoi)

    def fit(self, labels: List[str]):
        raise NotImplementedError()

    def tokenize(self, label: str) -> List[int]:
        raise NotImplementedError()

    def reverse_tokenize(self, idxs: List[int]) -> str:
        raise NotImplementedError()


class SelfiesTokenizer(Tokenizer):

    def fit(self, labels: List[str]):
        logging.info("Constructing SELFIES vocabulary")
        vocabulary = selfies.get_alphabet_from_selfies(tqdm(labels))
        vocabulary = sorted(list(vocabulary))
        vocabulary.extend(self.aux_tokens)
        self.stoi = {s: i for i, s in enumerate(vocabulary)}
        self.itos = dict(enumerate(vocabulary))

    def tokenize(self, label: str) -> List[int]:
        unk_idx = self.stoi['<UNK>']
        tokens = selfies.split_selfies(label)
        idxs = [self.stoi.get(token, unk_idx) for token in tokens]
        return idxs

    def reverse_tokenize(self, idxs: List[int], filter_aux: bool = True) -> str:
        tokens = map(self.itos.get, idxs)
        if filter_aux:
            tokens = filter(lambda t: t not in self.aux_tokens, tokens)
        return ''.join(tokens)


class MolecularCaptioningDataset(Dataset):

    def __init__(
            self,
            dataset_path: str,
            df_targets: pd.DataFrame,
            tokenizer: Tokenizer,
            transforms
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.df_targets = df_targets
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __len__(self):
        return self.df_targets.shape[0]

    def __getitem__(self, idx):
        row = self.df_targets.iloc[idx]
        image_id, target = row["image_id"], row["target"]
        img_path = get_img_file_path(
            image_id,
            "train",
            dataset_path=self.dataset_path
        )
        # img = torchvision.io.read_image(img_path)
        # img = img / 255.0
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = self.transforms(img)
        tgt_sequence = self.tokenizer.tokenize(target)
        tgt_len = torch.LongTensor([len(tgt_sequence)])
        tgt_sequence = torch.LongTensor(tgt_sequence)
        return img, tgt_sequence, tgt_len

    def collate_fn(self, batch):
        imgs, labels, label_lengths = [], [], []
        for data_point in batch:
            imgs.append(data_point[0])
            labels.append(data_point[1])
            label_lengths.append(data_point[2])
        labels = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.stoi["<PAD>"])
        return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)


class MolecularCaptioningValDataset(MolecularCaptioningDataset):

    def __getitem__(self, idx):
        img, tgt_sequence, tgt_len = super(MolecularCaptioningValDataset, self).__getitem__(idx)
        return img, tgt_sequence, tgt_len, torch.LongTensor([idx])

    def collate_fn(self, batch):
        imgs, labels, label_lengths = super(MolecularCaptioningValDataset, self).collate_fn(batch)
        idxs = []
        for data_point in batch:
            idxs.append(data_point[-1])
        return imgs, labels, label_lengths, torch.stack(idxs).reshape(-1, 1)

    def get_original_targets(self, idxs):
        return self.df_targets.iloc[idxs]["InChI"]


class MolecularCaptioningTestDataset(Dataset):

    def __init__(
            self,
            dataset_path: str,
            df_image_ids: pd.DataFrame,
            transforms
    ):
        self.dataset_path = dataset_path
        self.df_image_ids = df_image_ids
        self.transforms = transforms
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len_(self):
        return self.df_image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = self.df_image_ids.iloc[idx]['image_id']
        img_path = get_img_file_path(
            image_id,
            "test",
            dataset_path=self.dataset_path
        )
        # img = torchvision.io.read_image(img_path)
        # img = img / 255.0
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        _, h, w = img.size()
        if h > w:
            img = self.fix_transform(image=img)['image']
        img = self.transforms(img)
        return img


class MolecularCaptioningDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_path: str,
            target: str,
            targets_path: str,
            test_ids_path: str,
            imsize: int,
            batch_size: int,
            num_workers: int = 4,
            train_size: float = 0.8
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.targets_path = targets_path
        self.imsize = imsize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.df_targets = load_targets(targets_path, target)
        self.df_test_ids = pd.read_csv(test_ids_path)
        self.tokenizer = SelfiesTokenizer()
        self.train, self.val, self.test = None, None, None
        self.df_train, self.df_val = None, None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            df_train, df_val = train_test_split(
                self.df_targets,
                train_size=self.train_size,
                shuffle=True,
                random_state=42
            )
            self.df_train, self.df_val = df_train, df_val
            self.tokenizer.fit(df_train['target'])
            self.train = MolecularCaptioningDataset(
                self.dataset_path,
                df_targets=df_train,
                tokenizer=self.tokenizer,
                transforms=self._init_tforms("train")
            )
            self.val = MolecularCaptioningValDataset(
                self.dataset_path,
                df_targets=df_val,
                tokenizer=self.tokenizer,
                transforms=self._init_tforms("test")
            )
        if stage == 'test' or stage is None:
            self.test = MolecularCaptioningTestDataset(
                self.dataset_path,
                df_image_ids=self.df_test_ids,
                transforms=self._init_tforms("test")
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.train.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=4,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=self.val.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1024,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def _init_tforms(self, stage: str):
        tforms = None
        if stage == 'train':
            tforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(2 * (self.imsize,)),
                transforms.Normalize(
                    mean=PRETRAINED_MEAN,
                    std=PRETRAINED_STD
                )
            ])
        elif stage == 'test':
            tforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(2 * (self.imsize,)),
                transforms.Normalize(
                    mean=PRETRAINED_TEST_MEAN,
                    std=PRETRAINED_TEST_STD
                )
            ])
        return tforms

    def _aug_tfroms(self):
        return transforms.Compose([
            transforms.RandomErasing(p=0.2),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomRotation(90)])
        ])
