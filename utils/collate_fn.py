"""
@Time : 2020/12/1114:26
@Auth : 周俊贤
@File ：collate_fn.py
@DESCRIPTION:

"""
import torch

def sup_collate_fn(batch):
    max_len = max([x['length'] for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch], dtype=torch.long)
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch], dtype=torch.long)
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch], dtype=torch.long)

    all_labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)

    return {"all_input_ids":all_input_ids,
            "all_token_type_ids":all_token_type_ids,
            "all_attention_mask":all_attention_mask,
            "all_labels":all_labels}

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