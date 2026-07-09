import torch
from lenet import LeNet

# 加载保存的完整状态
ckpt = torch.load("./results/model.pt", map_location="cpu")
# 打印文件里存了什么（epoch、准确率、模型权重、优化器）
print("文件包含键：", list(ckpt.keys()))

# 加载模型权重
model = LeNet(dropout=False)
model.load_state_dict(ckpt["model_state_dict"])

# 查看第一层卷积权重形状
print("conv1权重shape：", model.conv1.weight.shape)
# 打印某一层参数数值
print(model.conv1.weight[0][0])