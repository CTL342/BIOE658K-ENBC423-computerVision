"""
train_models.py
Authentic reproduction script adhering directly to Hou et al. (CVPR 2021):
- MobileNetV2 with Coordinate Attention (mbv2_ca.py, 16 modules)
- MobileNetV2 + SE (r=24)
- MobileNetV2 + CBAM (r=24)
- MobileNetV2 Baseline
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# =======================================================================
# 1. Official Coordinate Attention Implementation (from coordatt.py)
# =======================================================================

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        return identity * x_w * x_h


# =======================================================================
# 2. Benchmark Attention Modules: SE and CBAM
# =======================================================================

class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduction=24):
        super(SqueezeAndExcitation, self).__init__()
        mid_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=24):
        super(CBAM, self).__init__()
        mid_channels = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_scale = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * channel_scale
        return x * self.spatial(self.compress(x))


# =======================================================================
# 3. Model Architecture (Line-for-Line from original/mbv2_ca.py)
# =======================================================================

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, attention_type=None, reduction=32):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            # Authors' original: First block has NO attention module
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            layers = [
                # 1. 1x1 point-wise expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # 2. 3x3 depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]

            # 3. Attention Placement (post-depthwise, pre-projection)
            if attention_type == 'ca':
                layers.append(CoordAtt(hidden_dim, hidden_dim, groups=reduction))
            elif attention_type == 'se':
                layers.append(SqueezeAndExcitation(hidden_dim, reduction=reduction))
            elif attention_type == 'cbam':
                layers.append(CBAM(hidden_dim, reduction=reduction))

            # 4. 1x1 linear point-wise projection
            layers.extend([
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv(x)
        return x + y if self.identity else y


class MBV2_CA(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, attention_type=None, reduction=32):
        super(MBV2_CA, self).__init__()
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(
                    inp=input_channel,
                    oup=output_channel,
                    stride=s if i == 0 else 1,
                    expand_ratio=t,
                    attention_type=attention_type,
                    reduction=reduction
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Authors' official classifier: nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(output_channel, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# =======================================================================
# 4. Training Engine & Distributed Cluster Setup
# =======================================================================

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    else:
        device = torch.device("cpu")
        if world_size > 1:
            dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=rank)

    return rank, local_rank, world_size, device


def get_data_loaders(data_dir, batch_size, workers, distributed=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler
    )

    return train_loader, val_loader, train_sampler, len(train_dataset.classes)


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss, total_top1, samples = 0.0, 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            prec1 = accuracy(outputs, targets, topk=(1,))[0]

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += prec1.item() * bs
            samples += bs

    if dist.is_initialized():
        metrics = torch.tensor([total_loss, total_top1, samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        tot_loss, tot_top1, tot_samples = metrics.tolist()
        return tot_loss / tot_samples, tot_top1 / tot_samples

    return total_loss / samples, total_top1 / samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--attention", type=str, default="ca", choices=["none", "se", "cbam", "ca"])
    parser.add_argument("--reduction", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./experiments")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    is_distributed = world_size > 1

    train_loader, val_loader, train_sampler, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.workers, is_distributed
    )

    attn_arg = None if args.attention == "none" else args.attention
    model = MBV2_CA(
        num_classes=num_classes,
        width_mult=1.0,
        attention_type=attn_arg,
        reduction=args.reduction
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # 200 epochs in original paper; 150 for subset convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        if rank == 0:
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                state = {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                    'best_acc': best_acc,
                }
                torch.save(state, os.path.join(args.output_dir, f"mbv2_{args.attention}_best.pth"))
                print(f"Epoch [{epoch:03d}/{args.epochs:03d}] New Best Acc: {best_acc:.2f}%")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()