# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from bert_lora_multilabel_cls import BertLoraMultiLabelCls
from data_helper_lora import MultiClsDataSet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import matplotlib.pyplot as plt
import os
import warnings

# 忽略sklearn的警告
warnings.filterwarnings('ignore')

# 数据路径和超参数
train_path = "/mnt/workspace/multi_label_classification/data/train.json"
dev_path = "/mnt/workspace/multi_label_classification/data/dev.json"
test_path = "/mnt/workspace/multi_label_classification/data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "/mnt/workspace/multi_label_classification/model/multi_label_finbert_lora100w.bin"
class_num = 8
device = "cuda" 
# if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 64
max_len = 256
hidden_size = 768
epochs = 10

# 定义保存图表的目录
SAVE_PLOT_DIR = "./plots"
os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

# 加载数据集
train_dataset = MultiClsDataSet(train_path, max_len=max_len)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

def evaluate_metrics(y_true_tensor, y_pred_prob_tensor):
    y_pred = (y_pred_prob_tensor.cpu() > 0.5).int().numpy()
    y_true = y_true_tensor.cpu().numpy()
    metrics = {
        'subset_accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred)
    }
    return metrics

def plot_training_metrics(metrics_history, save_dir):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['dev_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['train_subset_accuracy'], label='Training Subset Accuracy')
    plt.plot(metrics_history['dev_subset_accuracy'], label='Validation Subset Accuracy')
    plt.title('Training and Validation Subset Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['train_f1_micro'], label='Training F1 Micro')
    plt.plot(metrics_history['dev_f1_micro'], label='Validation F1 Micro')
    plt.plot(metrics_history['train_f1_macro'], label='Training F1 Macro')
    plt.plot(metrics_history['dev_f1_macro'], label='Validation F1 Macro')
    plt.title('Training and Validation F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics_lora.png'))
    plt.close()

def train():
    model = BertLoraMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    dev_best_f1 = 0.0
    patience = 3
    no_improvement_count = 0
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
    for epoch in range(1, epochs + 1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                metrics = evaluate_metrics(labels, logits)
                print(f"Train epoch:{epoch} step:{i}  "
                      f"Subset Acc: {metrics['subset_accuracy']:.4f} "
                      f"F1 Micro: {metrics['f1_micro']:.4f} "
                      f"F1 Macro: {metrics['f1_macro']:.4f} "
                      f"Loss: {loss.item():.4f}")
        metrics_history['train_loss'].append(loss.item())
        metrics_history['train_subset_accuracy'].append(metrics['subset_accuracy'])
        metrics_history['train_f1_micro'].append(metrics['f1_micro'])
        metrics_history['train_f1_macro'].append(metrics['f1_macro'])
        dev_loss, dev_metrics = dev(model, dev_dataloader, criterion)
        print(f"\nDev epoch:{epoch} "
              f"Subset Acc: {dev_metrics['subset_accuracy']:.4f} "
              f"Precision Micro: {dev_metrics['precision_micro']:.4f} "
              f"Recall Micro: {dev_metrics['recall_micro']:.4f} "
              f"F1 Micro: {dev_metrics['f1_micro']:.4f} "
              f"F1 Macro: {dev_metrics['f1_macro']:.4f} "
              f"Hamming Loss: {dev_metrics['hamming_loss']:.4f} "
              f"Loss: {dev_loss:.4f}\n")
        metrics_history['dev_loss'].append(dev_loss)
        metrics_history['dev_subset_accuracy'].append(dev_metrics['subset_accuracy'])
        metrics_history['dev_f1_micro'].append(dev_metrics['f1_micro'])
        metrics_history['dev_f1_macro'].append(dev_metrics['f1_macro'])
        if dev_metrics['f1_micro'] > dev_best_f1:
            dev_best_f1 = dev_metrics['f1_micro']
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved new best model with F1 Micro: {dev_best_f1:.4f}\n")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Early stopping...")
            break
    test_metrics = test(save_model_path, test_path)
    print("\nTest Results:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")
    plot_training_metrics(metrics_history, SAVE_PLOT_DIR)

def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_probs.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_probs = torch.cat(pred_probs, dim=0)
    metrics = evaluate_metrics(true_labels, pred_probs)
    return np.mean(all_loss), metrics

def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BertLoraMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_probs = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            true_labels.append(labels)
            pred_probs.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_probs = torch.cat(pred_probs, dim=0)
    return evaluate_metrics(true_labels, pred_probs)

if __name__ == '__main__':
    train()