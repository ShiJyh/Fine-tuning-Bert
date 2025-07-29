import json
import os

# 指定文件路径
file_path = '/mnt/workspace/multi_label_classification/data/train.json'

# 读取JSON文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤掉categories全为0的条目
filtered_data = [item for item in data if any(cat != 0 for cat in item.get('categories', []))]

# 将过滤后的数据写回原文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f'已移除 {len(data) - len(filtered_data)} 条全为0的条目。剩余 {len(filtered_data)} 条。')