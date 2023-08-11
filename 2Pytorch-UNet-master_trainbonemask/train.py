import pdb

from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from timm.scheduler import CosineLRScheduler


def train_net(net, device, data_path, epochs=5000, batch_size=16, lr=0.01):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-5)
    scheduler = CosineLRScheduler(optimizer, epochs * len(train_loader), lr_min=1e-7, warmup_lr_init=lr * 0.01,
                                  warmup_t=20 * len(train_loader), t_in_epochs=False)
    # 定义Loss算法
    criterion = nn.L1Loss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    begin_iter = 0
    for epoch in range(epochs):
        # 训练模式
        net.train()
        loss_sum = 0.0
        # 按照batch_size开始训练
        tau = 1.
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 得到一个二值化的mask label
            
            
            # 使用网络参数，输出预测结果
            pred = torch.sigmoid(net(image)/tau)
  
            # pdb.set_trace()
            # 计算loss
            loss = criterion(pred.view(pred.shape[0],-1), label.view(pred.shape[0],-1))
            
            

            print(f'epoch: {epoch} of {epochs}, lr: {scheduler._get_lr(begin_iter)[0]}, Loss/train: {loss.item()}')
            # 保存loss值最小的网络参数
            # 更新参数
            loss_sum += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step(begin_iter)
            begin_iter+=1
        print(f"total loss: {loss_sum}")
        if loss_sum < best_loss:
            best_loss = loss_sum
            torch.save(net.state_dict(), 'best_model.pth')
            print(f"save new dict")
        if epoch % 500 == 0:
            torch.save(net.state_dict(), f'{epoch} {loss_sum}.pth')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 两个unet
    
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "./train/"
    train_net(net, device, data_path)