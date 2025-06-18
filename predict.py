from newmodel import PSPNet
import torch
from PIL import Image
from  dataloader import VOCDataset
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
weight_path = 'weights/checkpoint--epoch50.pth'  # 调用训练得到的权重文件
img_file = 'cityscapes/aachen_000000_000019_gtFine_labelTrainIds.jpg'  # 进行预测的jpg图片
output = 'pred_output'  # 保存路径
model = PSPNet(num_classes=20)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(weight_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
MEAN = [0.28802046, 0.3264235, 0.28579238] # 数据集的均值和方差
STD = [0.1874096, 0.19018781, 0.1871926]
normalize = transforms.Normalize(MEAN, STD) #拿出训练时候dataloader 里面的设置
to_tensor = transforms.ToTensor()
model.to(device)
model.eval()

palette = [
    (128, 64, 128),    # road
    (244, 35, 232),    # sidewalk
    (70, 70, 70),      # building
    (102, 102, 156),   # wall
    (190, 153, 153),   # fence
    (153, 153, 153),   # pole
    (250, 170, 30),    # traffic light
    (220, 220, 0),     # traffic sign
    (107, 142, 35),    # vegetation
    (152, 251, 152),   # terrain
    (70, 130, 180),    # sky
    (220, 20, 60),     # person
    (255, 0, 0),       # rider
    (0, 0, 142),       # car
    (0, 0, 70),        # truck
    (0, 60, 100),      # bus
    (0, 80, 100),      # train
    (0, 0, 230),       # motorcycle
    (119, 11, 32),     # bicycle
    (0, 0, 0)          # ignored
]

def save_images(image, mask, output_path, image_file, palette,num_classes):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    image_file = os.path.basename(image_file).split('.')[0] # basenmae,返回图片名字
    colorized_mask = cam_mask(mask,palette,num_classes)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))

def cam_mask(mask,palette,n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask

with torch.no_grad():
    image = Image.open(img_file).convert('RGB')
    input = normalize(to_tensor(image)).unsqueeze(0)
    prediction = model(input.to(device))
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
    # =================================================
    save_images(image, prediction, output, img_file, palette,  num_classes=20)
