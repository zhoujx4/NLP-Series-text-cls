import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, BertConfig, AdamW

class CLS_model(nn.Module):
    def __init__(self, pretrained_model_path, embedding_dim, target_size):
        super(CLS_model, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrained_model_path)
        # self.tokenizer  = BertTokenizer.from_pretrained('./prev_trained_model/roberta/')
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.fc1 = nn.Linear(768*2, embedding_dim)
        self.activation1 = nn.Tanh()
        self.fc2 = nn.Linear(embedding_dim, target_size)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask = attention_mask)
        last_hidden_state = output[0]
        pooled_output = output[1]
        seq_avg = torch.mean(last_hidden_state, dim=1)
        seq_max = torch.max(last_hidden_state, dim=1)[0]
        concat_out = torch.cat((seq_avg, seq_max), dim=1)
        fc1_out = self.activation1(self.fc1(concat_out))
        fc2_out = self.fc2(fc1_out)

        return fc2_out