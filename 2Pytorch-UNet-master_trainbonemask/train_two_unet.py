import pdb

from model.two_unet_model import UNet
from utils.two_dataset import ISBI_Loader, split_train_and_val
from torch import optim
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch, math, os, sys
from timm.scheduler import CosineLRScheduler
import argparse
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils.save_image import change_tensor_to_image
parser = argparse.ArgumentParser(description="TWO STAGE UNET")
parser.add_argument(
    "--cuda_devices",
    type=str,
    default="0",
    help="data parallel training",
)
args = parser.parse_args()


def dice_loss(image_pred, image_gt, eps=1e-5):
    r""" computational formula�?
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    N = image_gt.size(0)
    pred_flat = image_pred.view(N, -1)
    gt_flat = image_gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.mean()

@torch.no_grad()
def test(net, dataloader):
    net.eval()
    total_sample = 0
    total_psnr = 0
    total_dice = 0

    def psnr(image_pred, image_gt, reduction="mean"):
        print("mse loss:", F.mse_loss(image_pred, image_gt, reduction=reduction))
        return -10 * torch.log(F.mse_loss(image_pred, image_gt, reduction=reduction)) / math.log(10)

    def dice(image_pred, image_gt, eps=1e-5):
        r""" computational formula�?
            dice = (2 * tp) / (2 * tp + fp + fn)
        """
        N = image_gt.size(0)
        pred_flat = image_pred.view(N, -1)
        gt_flat = image_gt.view(N, -1)

        tp = torch.sum(gt_flat * pred_flat, dim=1)
        fp = torch.sum(pred_flat, dim=1) - tp
        fn = torch.sum(gt_flat, dim=1) - tp
        loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return loss.sum() / N

    for i, (data, label, mask) in enumerate(dataloader):
        data, label,mask = data.cuda() - 0.5, label.cuda(), (mask>0).float().cuda()
        pred_mask,pred = net(data)
        pred = torch.sigmoid(pred)
        pred_mask = torch.sigmoid(pred_mask/0.05)
        _psnr = psnr(pred, label)
        _dice = dice(pred_mask,mask)
        total_sample += data.shape[0]
        total_psnr += (_psnr.item() * data.shape[0])
        total_dice += (_dice.item() * data.shape[0])

    return total_psnr / total_sample,total_dice/total_sample


def train_net(gpu, net, data_path, epochs=2000, batch_size=64, lr=0.02, accumulate_step=1):
    # 加载训练�?
    scaler = torch.cuda.amp.GradScaler()
    isbi_dataset = ISBI_Loader(data_path)  # 32,64,16
    train_dataset, val_dataset = split_train_and_val(isbi_dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               sampler=sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=False)
    # 定义RMSprop算法
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = CosineLRScheduler(optimizer, epochs * len(train_loader), lr_min=1e-7, warmup_lr_init=lr * 0.01,
                                  warmup_t=20 * len(train_loader), t_in_epochs=False)
    # 定义Loss算法
    criterion = nn.L1Loss()
    # best_loss统计，初始化为正无穷
    best_psnr = 0
    # 训练epochs�?
    begin_iter = 0
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_loader.sampler.set_epoch(epoch)
        loss_sum = 0.0
        # 按照batch_size开始训�?
        for i, (image, label, label_mask) in enumerate(train_loader):
            if begin_iter % accumulate_step == 0:
                optimizer.zero_grad()
            # 将数据拷贝到device�?
            image = image.cuda(gpu).float() - 0.5
            label = label.cuda(gpu).float()
            learnable_mask = (label.mean(dim=[1,2,3]) != -1)
            learnable_mask = learnable_mask.view(learnable_mask.shape[0],1,1,1).expand_as(image)
            label_mask = (label_mask.cuda(gpu)>0).float()
            if learnable_mask.sum().item()==0:
                continue
            with torch.cuda.amp.autocast(enabled=True):
                mask, logit = net(image)
                mask = torch.sigmoid(mask / 0.05)
                logit = torch.sigmoid(logit)

            # pdb.set_trace()
            # 计算loss

            loss1 = criterion(logit.view(label.shape[0], -1)[learnable_mask], label.view(label.shape[0], -1)[learnable_mask])
            loss2 = criterion(mask.view(label.shape[0], -1)[learnable_mask], label_mask.view(label.shape[0], -1)[learnable_mask])
            indice = label_mask.bool()
            loss3 = F.l1_loss(logit[indice&learnable_mask].float(),label[indice&learnable_mask].float(),reduction="mean")
            loss4 = 1 - dice_loss(mask[learnable_mask].float(),label_mask[learnable_mask].float())
            loss = loss1 + loss2 + loss3 + loss4 * 0.5
            if gpu==0:
                if begin_iter%100 == 0:
                    change_tensor_to_image(label[0].float(),"./",f"label_{begin_iter}")
                    change_tensor_to_image(mask[0].float(),"./",f"pre_mask_{begin_iter}")
                    change_tensor_to_image(logit[0].float(),"./",f"pre_label_{begin_iter}")
                    change_tensor_to_image(label_mask[0].float(),"./",f"mask_{begin_iter}")
            if gpu == 0:
                print(f'epoch: {epoch} of {epochs}, lr: {scheduler._get_lr(begin_iter)[0]}, Loss1/train: {loss1.item()}, Loss2/train: {loss2.item()}, Loss3/train: {loss3.item()}, Loss4/train: {loss4.item()}')
            # 保存loss值最小的网络参数
            # 更新参数
            loss_sum += loss.item()
            scaler.scale(loss / accumulate_step).backward()
            if (begin_iter + 1) % accumulate_step == 0:
                scaler.step(optimizer)
                scaler.update()
            scheduler.step(begin_iter)
            begin_iter += 1
        if gpu == 0:
            print(f"total train loss: {loss_sum}")
            _psnr,_dice = test(net, val_loader)
            print(f"total val psnr: {_psnr}, total val dice: {_dice}")
            if _psnr+2*_dice > best_psnr:
                best_psnr = _psnr+2*_dice
                torch.save(net.state_dict(), f'two_unet_best_model_both_f_and_t_bone+.pth')
                print(f"save new dict")
            # if epoch % 500 == 0:
            #     torch.save(net.state_dict(), f'two_unet {epoch} {loss_sum}.pth')


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def set_random_seed(number=0):
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import random
    import numpy as np
    np.random.seed(number)
    random.seed(number)


def main_worker(gpu, ngpus_per_node, world_size, dist_url):
    print("Use GPU: {} for training".format(gpu))
    rank = 0  # 单机
    dist_backend = "nccl"
    rank = rank * ngpus_per_node + gpu
    print("world_size:", world_size)
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )
    set_random_seed(rank + np.random.randint(0, 1000))
    torch.cuda.set_device(gpu)
    net = DDP(UNet(n_channels=1, n_classes=1).cuda(gpu), device_ids=[gpu])
    # 指定训练集地址，开始训�?
    # data_path = "./train/"
    data_path = "/data1/hz/data_hz/unet/data/data/train/"
    batchsize = 48
    initial_lr = 0.015
    node_batchsize = batchsize // ngpus_per_node
    initial_lr = (initial_lr / 4) * ngpus_per_node
    train_net(gpu, net, data_path, batch_size=node_batchsize, lr=initial_lr)
    dist.destroy_process_group()


if __name__ == "__main__":
    device = set_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["MASTER_ADDR"] = "127.0.0.1"  #
    os.environ["MASTER_PORT"] = "8888"  #
    world_size = 1
    port_id = 10002 + np.random.randint(0, 1000) + int(args.cuda_devices[0])
    dist_url = "tcp://127.0.0.1:" + str(port_id)
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * world_size
    print("multiprocessing_distributed")
    torch.multiprocessing.set_start_method("spawn")
    mp.spawn(  # Left 2: softmax weight=1 Right 2: softmax weight=2
        main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, world_size, dist_url)
    )

"""
L1LOSS,Adam:
total train loss: 1.6117717251181602
total val psnr: 24.768162476389033
L1LOSS,Adam,bs=16,epoch=2000:
total train loss: 0.4765358120203018
total val psnr: 25.88375553331877
L1LOSS,Adam,bs=32,epoch=2000:
total train loss: 0.5027387719601393                                                                                          │····················�?
total val psnr: 25.55936040376362
L1LOSS,Adam,bs=64,epoch=2000:
total train loss: 0.5666997376829386
total val psnr: 25.275752720079925
L1LOSS,AdamW:
total train loss: 1.6121831312775612
total val psnr: 24.674584740086605
MSELOSS:
total train loss: 0.6421697195619345                                                                                                                
total val psnr: 23.060366781134356   
"""
