# PSP-CBMA城市街景语义分割

## 数据集
数据来源：https://www.cityscapes-dataset.com/downloads/

gtFine_trainvaltest.zip (241MB) 

leftImg8bit_trainvaltest.zip (11GB)

### 数据处理

按照cityscapes官方文件处理脚本https://github.com/mcordts/cityscapesScripts
对于cityscapesscripts/helpers/label.py中从id到trainId映射关系结合json文件处理png标签图片

并将图片的格式设置整理为VOC数据集格式，使用voc_annotation.py划分训练集和测试集生成对应的txt文件

    项目根目录/
    ├── cityscapes/
    │   ├── train.txt
    │   ├── valid.txt
    │   └── JPEGImages/
    │       └── a.jpg
    │   └── mask/
    │       └── a.png



## 参考
代码参考：https://blog.csdn.net/weixin_47142735/article/details/116857782



