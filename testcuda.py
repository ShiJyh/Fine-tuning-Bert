# import torch
# print(torch.version.cuda)        # PyTorch 编译时绑定的 CUDA 版本
# print(torch.cuda.is_available()) # 先看 GPU 是否可用

import torch
print("PyTorch:", torch.__version__)      # 2.1.0+cu118
# print("CUDA build:", torch.version.cuda)  # 11.8
# print("Driver >=", torch.cuda.get_device_capability())