# -*- coding: utf-8 -*-

# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from peft import get_peft_model, LoraConfig

# 定义多标签分类模型，使用LoRA微调Bert
class BertLoraMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, class_num, dropout=0.1):
        super(BertLoraMultiLabelCls, self).__init__()
        # 加载预训练的Bert模型
        self.bert = BertModel.from_pretrained("/mnt/workspace/finbert1")
        # 配置LoRA参数：r是LoRA的秩，lora_alpha是缩放因子，target_modules指定应用LoRA的模块
        lora_config = LoraConfig(
            r=16,  # LoRA的低秩维度
            lora_alpha=32,  # LoRA缩放因子
            target_modules=["query", "value"],  # 应用LoRA的注意力模块
            lora_dropout=0.1,  # dropout率
            bias="none",  # 不调整偏置
        )
        # 将Bert模型转换为LoRA模型，只微调LoRA参数
        self.bert = get_peft_model(self.bert, lora_config)
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size, class_num)
        # dropout层防止过拟合
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 前向传播：通过LoRA调整的Bert模型
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取池化输出并应用dropout
        cls = self.drop(outputs.pooler_output)
        # 通过全连接层并应用sigmoid激活，用于多标签概率输出
        out = F.sigmoid(self.fc(cls))
        return out