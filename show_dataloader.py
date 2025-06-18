## 数据处理展示

import matplotlib.pyplot as plt
import numpy as np
from dataloader import VOCDataset  # 导入你的数据集类

def visualize_samples(dataset, indices):
    # 创建一个2x4的子图布局
    fig, axes = plt.subplots(2, 4, figsize=(8,4))

    for i, index in enumerate(indices):
        # 获取样本
        image_tensor, label_tensor = dataset[index]

        # 转换图像张量到可显示格式
        image = image_tensor.numpy().transpose(1, 2, 0)  # CHW -> HWC
        image = image * np.array(dataset.STD) + np.array(dataset.MEAN)  # 反归一化
        image = np.clip(image, 0, 1)  # 确保值在[0,1]范围内

        # 准备标签
        label = label_tensor.numpy()

        # 计算子图的位置
        row = i // 2
        col = (i % 2) * 2

        # 显示原始图像
        axes[row, col].imshow(image)
        axes[row, col].axis('off')

        # 显示标签掩码
        axes[row, col + 1].imshow(label, cmap='jet', vmin=0, vmax=dataset.num_classes)
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.savefig('dl.png')
    plt.show()

if __name__ == "__main__":
    dataset = VOCDataset(
        root='cityscapes',
        split='train',
        num_classes=20,
        base_size=713,
        augment=True,
        crop_size=713
    )

    print(f"数据集大小: {len(dataset)}")

    # 选择前4个样本的索引
    indices = np.arange(4)

    # 可视化4组样本
    visualize_samples(dataset, indices)
