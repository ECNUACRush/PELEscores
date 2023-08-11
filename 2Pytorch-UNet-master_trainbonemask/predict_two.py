import copy
import glob
import pdb

import numpy as np
import torch
import os
import cv2
import torchvision.transforms
from PIL import Image
# from model.unet_model import UNet
from model.two_unet_model import UNet
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数

    # net.load_state_dict(torch.load('best_model.pth', map_location=device))
    dict = torch.load('two_unet_best_model_both_f_and_t_bone.pth', map_location=device)
    new_dict = {}
    for key, value in dict.items():
        new_key = key.replace("module.", "")
        new_dict[new_key] = value
    net.load_state_dict(new_dict)

    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob(
        '/data1/hz/data_hz/cycleResult/384_resnet9layer_260/*.png')
    # 遍历所有图片
    # pdb.set_trace()
    outputs_path = "/data1/hz/data_hz/test/384_resnet9layer_260/"
    if not os.path.exists(outputs_path+"mask"):
        os.makedirs(outputs_path+"mask")
    if not os.path.exists(outputs_path+"pred"):
        os.makedirs(outputs_path+"pred")
    with torch.no_grad():
        for test_path in tests_path:
            # 保存结果地址
            # 读取图片
            img = Image.open(test_path).convert("L")
            img = torchvision.transforms.Resize([384,384])(img)
            img = torchvision.transforms.ToTensor()(img)
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[1], img.shape[2])
            img_tensor = img.to(device=device, dtype=torch.float32) - 0.5
            # pred =(net(img_tensor)).sigmoid()
            # pdb.set_trace()
            mask, pred = net(img_tensor)
            mask, pred = (mask / 0.05).sigmoid(), (pred / 1).sigmoid()

            # pdb.set_trace()

            # print(pred)

            # pdb.set_trace()
            # 提取结果
            # [1, 384, 384] => [384, 384]
            pred = np.array(pred.data.cpu()[0])[0]

            mask = np.array(mask.data.cpu()[0])[0]

            origin = np.array(img_tensor.data.cpu()[0])[0]

            # pdb.set_trace()
            # 处理结果
            #
            # pred[pred < 30/255] = 0
            pred *= 255
            mask *= 255
            origin += 0.5
            origin *= 255
            pred += 0.5
            mask += 0.5
            origin += 0.5
            pred = np.clip(pred, 0, 255)
            mask = np.clip(mask, 0, 255)
            origin = np.clip(origin,0,255)

            pred = pred.astype(np.uint8)

            # mask[mask < 30/255] = 0
            mask = mask.astype(np.uint8)
            origin = origin.astype(np.uint8)
            # pred = (pred - pred.min())/(pred.max() - pred.min()) * 255

            # unique, counts = np.unique(pred, return_counts=True)
            # print(dict(zip(unique, counts)))
            # pred[pred < 10] = 0
            # pdb.set_trace()
            # unique, counts = np.unique(pred, return_counts=True)
            # print(dict(zip(unique, counts)))

            # pred = (pred - pred.min()) / (pred.max() - pred.min())

            # 保存图片
            output_mask_path = outputs_path + "mask" + test_path[test_path.rfind("/"):]
            output_pred_path = outputs_path + "pred" + test_path[test_path.rfind("/"):]
            print(output_pred_path,output_mask_path)
            cv2.imwrite(output_mask_path, mask)
            cv2.imwrite(output_pred_path, pred)


