import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# 配置路径和类别
base_dir = "SUCCESS/draw"  # 替换为你的图片根目录
categories = ['Real','GroundTruth','res50_PPM', 'res50_PPM_CBAM']

# 创建画布，设置宽高比为1:2
fig = plt.figure(figsize=(12, 6))  # 高度是宽度的2倍
gs = GridSpec(4, 4, figure=fig, 
              wspace=0.015, hspace=0.01,  # 几乎无间距
            #   left=0.02, right=0.98,
            #   top=0.98, bottom=0.02)
                )

# 遍历每个类别（列）
for col_idx, category in enumerate(categories):
    folder_path = os.path.join(base_dir, category)
    images = sorted([f for f in os.listdir(folder_path) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 遍历每张图片（行）
    for row_idx, img_name in enumerate(images):
        img_path = os.path.join(folder_path, img_name)
        img = mpimg.imread(img_path)
        
        # 创建子图并显示图片
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(img)
        ax.axis('off')  # 关闭坐标轴
        
        # 只在第一行添加列标题
        if row_idx == 0:
            ax.set_title(f"{category}", fontsize=8, pad=6)

plt.savefig("image_matrix.png", bbox_inches='tight', dpi=300)  # 提高DPI使图片更清晰
plt.show()
