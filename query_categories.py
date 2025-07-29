import json

# 指定文件路径
file_path = '/mnt/workspace/multi_label_classification/data/test.json'

# 读取JSON文件
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化计数器
zero_count = 0
category_counts = [0] * 8  # 假设有8个类别，根据数据调整

# 遍历数据
for item in data:
    categories = item.get('categories', [])
    if all(cat == 0 for cat in categories):
        zero_count += 1
    for i, cat in enumerate(categories):
        if cat == 1:
            category_counts[i] += 1

# 输出结果
print(f'categories全为0的条目数量: {zero_count}')
print('每个类别的数量:')
for i, count in enumerate(category_counts):
    print(f'类别 {i+1}: {count}')