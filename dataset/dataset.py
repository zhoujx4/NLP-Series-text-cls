"""
@Time : 2020/12/1114:22
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION: define dataset

"""

import random

import torch
from nlpcda import Similarword
from torch.utils.data import Dataset
import pandas as pd

class CLSDataset(Dataset):
    def __init__(self, args, csv_path, tokenizer, type="train", is_augement=False):
        '''
        cat should in ["客体类别", "销售方式", "举报问题类别", "投诉问题类别"]
        '''
        self.df = pd.read_csv(csv_path)
        # if type == "train":
            # self.df = self.df.sample(100).reset_index(drop=True)
        self.max_seq_length = args.train_max_seq_length
        self.tokenizer = tokenizer
        self.type = type
        self.is_augement = is_augement
        if self.is_augement:
            self.smw = Similarword(create_num=2, change_rate=0.2)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        content = self.df.loc[index, "sentence"]
        if self.type == "train" and self.is_augement:
            if random.random() > 0.5:
                content = self.smw.replace(content)[-1]
        label = self.df.loc[index, "label"]
        # d_encode = self.tokenizer.encode_plus(content)
        # padding
        d_encode = self.tokenizer.encode_plus(content,
                                              padding="max_length",
                                              max_length=self.max_seq_length,
                                              truncation=True)

        sample = {"input_ids": d_encode['input_ids'],
            "token_type_ids": d_encode['token_type_ids'],
            "attention_mask": d_encode['attention_mask'],
            "length" : sum(d_encode['attention_mask']),
            "label": label}
        return sample

class UDADataset(Dataset):
    def __init__(self, args, csv_path, tokenizer):
        '''
        cat should in ["客体类别", "销售方式", "举报问题类别", "投诉问题类别"]
        '''
        self.df = pd.read_csv(csv_path)
        self.max_seq_length = args.train_max_seq_length
        self.smw = Similarword(create_num=2, change_rate=0.3)
        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):
        content = self.df.loc[index, "sentence"]
        contents = self.smw.replace(content)
        if len(contents) == 1:
            contents.append(contents[0])
        sample = {}
        for index, content in enumerate(contents):
            d_encode = self.tokenizer.encode_plus(content)
            # padding
            if self.max_seq_length >= len(d_encode['input_ids']):
                length = len(d_encode['input_ids'])
                padding_length = self.max_seq_length - len(d_encode['input_ids'])
                d_encode['input_ids'] += [self.tokenizer.pad_token_id] * padding_length
                d_encode['token_type_ids'] += [0] * padding_length
                d_encode['attention_mask'] += [0] * padding_length
            elif self.max_seq_length < len(d_encode['input_ids']):
                length = self.max_seq_length
                d_encode['input_ids'] = d_encode['input_ids'][:self.max_seq_length]
                d_encode['token_type_ids'] = d_encode['token_type_ids'][:self.max_seq_length]
                d_encode['attention_mask'] = d_encode['attention_mask'][:self.max_seq_length]

            assert len(d_encode['input_ids']) == self.max_seq_length
            assert len(d_encode['token_type_ids']) == self.max_seq_length
            assert len(d_encode['attention_mask']) == self.max_seq_length

            if index == 0:
                sample.update({
                    "ori_input_ids": d_encode['input_ids'],
                    "ori_token_type_ids": d_encode['token_type_ids'],
                    "ori_attention_mask": d_encode['attention_mask'],
                    "ori_length": length,
                })
            elif index == 1:
                sample.update({
                    "aug_input_ids": d_encode['input_ids'],
                    "aug_token_type_ids": d_encode['token_type_ids'],
                    "aug_attention_mask": d_encode['attention_mask'],
                    "aug_length": length,
                })
        return sample

def sup_collate_fn(batch):
    max_len = max([x['length'] for x in batch])
    input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    label = torch.tensor([x["label"] for x in batch])


    return {"all_input_ids":input_ids,
            "all_attention_mask":attention_mask,
            "all_token_type_ids":token_type_ids,
            "all_labels":label}

def unsup_collate_fn(batch):
    ori_length = [x['ori_length'] for x in batch]
    aug_length = [x['aug_length'] for x in batch]
    ori_length.extend(aug_length)
    max_len = max(ori_length)
    #
    ori_input_ids = torch.tensor([x['ori_input_ids'][:max_len] for x in batch], dtype=torch.long)
    aug_input_ids = torch.tensor([x['aug_input_ids'][:max_len] for x in batch], dtype=torch.long)
    ori_token_type_ids = torch.tensor([x['ori_token_type_ids'][:max_len] for x in batch], dtype=torch.long)
    aug_token_type_ids = torch.tensor([x['aug_token_type_ids'][:max_len] for x in batch], dtype=torch.long)
    ori_attention_mask = torch.tensor([x['ori_attention_mask'][:max_len] for x in batch], dtype=torch.long)
    aug_attention_mask = torch.tensor([x['aug_attention_mask'][:max_len] for x in batch], dtype=torch.long)
    #
    all_input_ids = torch.cat([ori_input_ids, aug_input_ids], dim=0)
    all_token_type_ids = torch.cat([ori_token_type_ids, aug_token_type_ids], dim=0)
    all_attention_mask = torch.cat([ori_attention_mask, aug_attention_mask], dim=0)
    return {"all_input_ids":all_input_ids,
            "all_token_type_ids":all_token_type_ids,
            "all_attention_mask":all_attention_mask}