# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_preprocess import load_json


class MultiClsDataSet(Dataset):
    def __init__(self, data_path, max_len=256, label2idx_path="./data/label2idx.json"):
        #self.label2idx = load_json(label2idx_path)
        self.class_num = 8
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/workspace/bert-base")
        self.max_len = max_len
        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.encoder(data_path)

    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path,encoding='utf-8') as f:
            data_list = json.load(f)
        for item in data_list:
            texts.append(item["text"])
            labels.append(item["categories"])
        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, token_type_ids, attention_mask, \
               torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input_ids[item],  self.attention_mask[item], \
               self.token_type_ids[item], self.labels[item]


if __name__ == '__main__':
    dataset = MultiClsDataSet(data_path="./data/train.json")
    print(dataset.input_ids)
    print(dataset.token_type_ids)
    print(dataset.attention_mask)
    print(dataset.labels)
