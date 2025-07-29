# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.bert = BertModel.from_pretrained("/mnt/workspace/finbert1")
        # #冻结BERT的所有参数（不更新梯度）
        # for param in self.bert.parameters():
        #     param.requires_grad = False  # 关键修改！


        # finbert 解冻策略
        # 冻结所有层
        for param in self.bert.parameters():
            param.requires_grad = False

        # 解冻最后6层（7-12层）和Pooler
        for layer in self.bert.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(hidden_size, class_num)
#         self.fc = nn.Sequential(
#     nn.Linear(hidden_size, 256),  # 中间层
#     nn.ReLU(),
#     nn.Linear(256, class_num)    # 输出层
# )
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        #单层情况下
        cls = self.drop(outputs[1])
        out = F.sigmoid(self.fc(cls))
        return out


        #  # BERT前向传播（仅推理，不计算梯度）
        # with torch.no_grad():  # 显式禁用梯度，提升效率
        #     outputs = self.bert(input_ids, attention_mask, token_type_ids)
        
        # # 获取[CLS]向量
        # cls_output = outputs[1]  # outputs[1]对应pooler_output
        
        # # 分类头前向传播
        # cls_output = self.drop(cls_output)
        # logits = self.fc(cls_output)
        # return torch.sigmoid(logits)  # 多标签分类用Sigmoid








