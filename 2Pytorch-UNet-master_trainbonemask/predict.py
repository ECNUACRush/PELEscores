import glob
import pdb

import numpy as np
import torch
import os
import cv2
import torchvision.transforms
from PIL import Image
# from model.unet_model import UNet
from model.unet_model import UNet

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1�?
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce�?
    net.to(device=device)
    # 加载模型参数
    
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    #net.load_state_dict(torch.load('best_model_old.pth', map_location=device))
    
    # 测试模式
    net.eval()
    # 读取所有图片路�?
    # tests_path = glob.glob('/data1/hz/data_hz/cycleResult/1024_resnet9layer_200/*.png')
    # tests_path = glob.glob('./test/*.png')
    tests_path = glob.glob('/data1/hz/data_hz/unet/data/data/train/train_store/image/*.png')
    
    num = 0
    # 遍历所有图�?
    # pdb.set_trace()
    for test_path in tests_path:
        num += 1
        if (num == 30):
          break
        # 保存结果地址
        # save_res_path = test_path.replace('cycleResult', 'Unet_answer') + '_res.png'
        save_res_path = test_path.replace('image', 'save') 
        # save_res_path = test_path + '_res.png'
        # pdb.set_trace()
        # 读取图片
        img = Image.open(test_path).convert("L").resize((1024,1024))
        # img_new = cv2.resize(img, (1024, 1024))
        img = torchvision.transforms.ToTensor()(img)
        print(img.shape)
        # 转为batch�?，通道�?，大小为512*512的数�?
        img = img.reshape(1, 1, img.shape[1], img.shape[2])
        img_tensor =img.to(device=device, dtype=torch.float32)
        pred =(net(img_tensor)).sigmoid()
        
        # mask =(net(img_tensor)[0]).sigmoid()
        # pred =(net(img_tensor)[0]).sigmoid()
        # pdb.set_trace()
        
        # print(pred)

        # pdb.set_trace()
        # 提取结果
        # [1, 384, 384] => [384, 384]
        pred = np.array(pred.data.cpu()[0])[0]
        
        # mask = np.array(mask.data.cpu()[0])[0]
        
        # pdb.set_trace()
        # 处理结果
        #
        pred[pred < 30/255] = 0
        pred *= 255
        # mask *= 255
        pred += 0.5
        pred = np.clip(pred,0,255)
        # mask = np.clip(pred, 0, 255)
        
        
        pred = pred.astype(np.uint8)
        
        # mask[mask < 30/255] = 0
        # mask = mask.astype(np.uint8)
        
        
     
        # pred = (pred - pred.min())/(pred.max() - pred.min()) * 255
        
        
        
        # unique, counts = np.unique(pred, return_counts=True)
        # print(dict(zip(unique, counts)))
        # pred[pred < 10] = 0
        # pdb.set_trace()
        # unique, counts = np.unique(pred, return_counts=True)
        # print(dict(zip(unique, counts)))
        
        # pred = (pred - pred.min()) / (pred.max() - pred.min())


        # 保存图片
        # cv2.imwrite(save_res_path, pred)
  
        cv2.imwrite(save_res_path, pred)
