# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os 

train_path = "/mnt/workspace/multi_label_classification/data/train.json"
dev_path = "/mnt/workspace/multi_label_classification/data/dev.json"
test_path = "/mnt/workspace/multi_label_classification/data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "/mnt/workspace/multi_label_classification/model/multi_label_cls_finbert_feeze_128_10.bin"
#label2idx = load_json(label2idx_path)
class_num = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 128
max_len = 256
hidden_size = 768
epochs = 10

#try1:
'''
1.数据质量问题
2.数据分布不均匀问题
方式： 1.mengzi bert 全微调  2.mengzi bert 分类头微调  3.finbert 全微调  4.finbert 分类头微调(BERT高层7-12层\Pooler层\自定义分类头)
'''

#mengzi finbert
#不冻结参数全调参 会过拟合 128_10   epoch:9 acc:0.3451733514615907 loss:0.3791661119979361  Test acc: 0.325

#冻结参数仅仅微调分类头  训练效果很差 不拟合

##finbert #数据目前可能不够多
# 不冻结参数全调参  128_10  仍然过拟合

#冻结部分参数 128_10 仍然过拟合
#Dev epoch:9 acc:0.31101291638341266 loss: 0.38423042064127716
#Test acc: 0.275s

#冻结部分参数 新数据集 128_10_0.1  2w训练集
'''
Test Results:
subset_accuracy: 0.3594
precision_micro: 0.7152
recall_micro: 0.6496
f1_micro: 0.6808
precision_macro: 0.6994
recall_macro: 0.6307
f1_macro: 0.6597
hamming_loss: 0.1354
'''

#冻结部分参数 新数据集 128_10_0.3  2w训练集

'''
Test Results:
subset_accuracy: 0.3560
precision_micro: 0.7105
recall_micro: 0.6507
f1_micro: 0.6793
precision_macro: 0.6933
recall_macro: 0.6358
f1_macro: 0.6623
hamming_loss: 0.1366
'''

##冻结全部参数 新数据集 128_5_0.1   2w训练集 很烂

##训练全参数

train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)


def get_acc_score(y_true_tensor, y_pred_tensor):
    y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_true_tensor = y_true_tensor.cpu().numpy()
    return accuracy_score(y_true_tensor, y_pred_tensor)


def train():
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num,dropout=0.3)
    for name, param in model.named_parameters():
        print(name, "可训练" if param.requires_grad else "冻结")
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dev_best_acc = 0.

    # 初始化指标记录列表
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': []
    }

    for epoch in range(1, epochs):
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
                acc_score = get_acc_score(labels, logits)
                print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))

        # 在每个epoch结束后记录指标
        metrics_history['train_loss'].append(loss.item())
        metrics_history['train_acc'].append(acc_score)

        # 验证集合
        dev_loss, dev_acc = dev(model, dev_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss: {}".format(epoch, dev_acc, dev_loss))
        metrics_history['dev_loss'].append(dev_loss)
        metrics_history['dev_acc'].append(dev_acc)

        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            torch.save(model.state_dict(), save_model_path)

    # 测试
    test_acc = test(save_model_path, test_path)
    print("Test acc: {}".format(test_acc))

    # 保存训练指标图
    plt.figure(figsize=(12, 5))

    # 绘制训练和验证的损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['dev_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和验证的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['train_acc'], label='Training Accuracy')
    plt.plot(metrics_history['dev_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PLOT_DIR, 'training_metrics.png'))
    plt.close()


def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return acc_score


if __name__ == '__main__':
    train()
