# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import matplotlib.pyplot as plt
import os
import warnings

# 忽略sklearn的警告
warnings.filterwarnings('ignore')

train_path = "/mnt/workspace/multi_label_classification/data/train.json"
dev_path = "/mnt/workspace/multi_label_classification/data/dev.json"
test_path = "/mnt/workspace/multi_label_classification/data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "/mnt/workspace/multi_label_classification/model/multi_label_bertbase_cls_f1_new_data_feezeall.bin"
class_num = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 128
max_len = 256
hidden_size = 768
epochs = 10

# 定义保存图表的目录
SAVE_PLOT_DIR = "./plots"
os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

# 加载数据集
train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

def evaluate_metrics(y_true_tensor, y_pred_prob_tensor):
    """
    计算多标签分类的各种评估指标
    
    参数:
        y_true_tensor: 真实标签张量 (shape: [batch_size, num_labels])
        y_pred_prob_tensor: 预测概率张量 (shape: [batch_size, num_labels])
        
    返回:
        metrics: 包含各种评估指标的字典
    """
    # 将概率转换为二进制预测 (0.5为阈值)
    y_pred = (y_pred_prob_tensor.cpu() > 0.5).int().numpy()
    y_true = y_true_tensor.cpu().numpy()
    
    # 计算各种指标
    metrics = {
        # 子集准确率 (所有标签都预测正确才算正确)
        'subset_accuracy': accuracy_score(y_true, y_pred),
        
        # 微平均 (按样本-标签对计算)
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        
        # 宏平均 (每个标签单独计算后平均)
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        
        # Hamming损失 (错误标签的比例)
        'hamming_loss': hamming_loss(y_true, y_pred)
    }
    return metrics

def train():
    """模型训练函数"""
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 使用微平均F1分数作为早停标准
    dev_best_f1 = 0.0

    # 初始化指标记录字典
    metrics_history = {
        'train_loss': [],
        'train_subset_accuracy': [],
        'train_f1_micro': [],
        'train_f1_macro': [],
        'dev_loss': [],
        'dev_subset_accuracy': [],
        'dev_f1_micro': [],
        'dev_f1_macro': []
    }

    for epoch in range(1, epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            input_ids, attention_mask, token_type_ids, labels = batch
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 每10个batch打印一次训练信息
            if i % 10 == 0:
                metrics = evaluate_metrics(labels, logits)
                print(f"Train epoch:{epoch} step:{i}  "
                      f"Subset Acc: {metrics['subset_accuracy']:.4f} "
                      f"F1 Micro: {metrics['f1_micro']:.4f} "
                      f"F1 Macro: {metrics['f1_macro']:.4f} "
                      f"Loss: {loss.item():.4f}")

        # 记录训练指标
        metrics_history['train_loss'].append(loss.item())
        metrics_history['train_subset_accuracy'].append(metrics['subset_accuracy'])
        metrics_history['train_f1_micro'].append(metrics['f1_micro'])
        metrics_history['train_f1_macro'].append(metrics['f1_macro'])

        # 验证集评估
        dev_loss, dev_metrics = dev(model, dev_dataloader, criterion)
        print(f"\nDev epoch:{epoch} "
              f"Subset Acc: {dev_metrics['subset_accuracy']:.4f} "
              f"Precision Micro: {dev_metrics['precision_micro']:.4f} "
              f"Recall Micro: {dev_metrics['recall_micro']:.4f} "
              f"F1 Micro: {dev_metrics['f1_micro']:.4f} "
              f"F1 Macro: {dev_metrics['f1_macro']:.4f} "
              f"Hamming Loss: {dev_metrics['hamming_loss']:.4f} "
              f"Loss: {dev_loss:.4f}\n")

        # 记录验证指标
        metrics_history['dev_loss'].append(dev_loss)
        metrics_history['dev_subset_accuracy'].append(dev_metrics['subset_accuracy'])
        metrics_history['dev_f1_micro'].append(dev_metrics['f1_micro'])
        metrics_history['dev_f1_macro'].append(dev_metrics['f1_macro'])

        # 使用微平均F1分数作为模型保存标准
        if dev_metrics['f1_micro'] > dev_best_f1:
            dev_best_f1 = dev_metrics['f1_micro']
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved new best model with F1 Micro: {dev_best_f1:.4f}\n")

    # 测试集评估
    test_metrics = test(save_model_path, test_path)
    print("\nTest Results:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    # 绘制并保存训练指标图
    plot_training_metrics(metrics_history, SAVE_PLOT_DIR)


def plot_training_metrics(metrics_history, save_dir):
    """绘制并保存训练指标图"""
    plt.figure(figsize=(15, 6))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['dev_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制子集准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['train_subset_accuracy'], label='Training Subset Accuracy')
    plt.plot(metrics_history['dev_subset_accuracy'], label='Validation Subset Accuracy')
    plt.title('Training and Validation Subset Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制F1分数曲线 (微平均和宏平均)
    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['train_f1_micro'], label='Training F1 Micro')
    plt.plot(metrics_history['dev_f1_micro'], label='Validation F1 Micro')
    plt.plot(metrics_history['train_f1_macro'], label='Training F1 Macro')
    plt.plot(metrics_history['dev_f1_macro'], label='Validation F1 Macro')
    plt.title('Training and Validation F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics_f1_0.1_bertbase.png'))
    plt.close()


def dev(model, dataloader, criterion):
    """验证集评估函数"""
    all_loss = []
    model.eval()
    true_labels = []
    pred_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_probs.append(logits)
    
    # 合并所有批次的结果
    true_labels = torch.cat(true_labels, dim=0)
    pred_probs = torch.cat(pred_probs, dim=0)
    
    # 计算评估指标
    metrics = evaluate_metrics(true_labels, pred_probs)
    return np.mean(all_loss), metrics


def test(model_path, test_data_path):
    """测试集评估函数"""
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    true_labels = []
    pred_probs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids, attention_mask, token_type_ids)
            true_labels.append(labels)
            pred_probs.append(logits)
    
    # 合并所有批次的结果
    true_labels = torch.cat(true_labels, dim=0)
    pred_probs = torch.cat(pred_probs, dim=0)
    
    # 计算评估指标
    return evaluate_metrics(true_labels, pred_probs)


if __name__ == '__main__':
    train()