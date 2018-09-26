# -*- coding: utf-8 -*-

"""
QSegNet train routines. (WIP)
"""

# Standard lib imports
import os
import time
import argparse
import os.path as osp
# import multiprocessing
from urllib.parse import urlparse
import pdb
# multiprocessing.set_start_method("spawn", force=True)

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from utils_ed import AverageMeter
from psp import PSPNet
from data_loader2 import LiverDataset

# Other imports
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='AngelaNet version Liver')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='for_FCN',
                    help='path to ReferIt splits data folder')
parser.add_argument('--split-root', type=str, default='data2',
                    help='path to dataloader splits data folder')
parser.add_argument('--save-folder', default='weights/',
                    help='location to save checkpoint models')
parser.add_argument('--snapshot', default='weights/angela_snapshot.pth',
                    help='path to weight snapshot file')
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of workers used in dataloading')
parser.add_argument('--split', default='train', type=str,
                    help='name of the dataset split used to train')
parser.add_argument('--val', default='test', type=str,
                    help='name of the dataset split used to validate')
parser.add_argument('--eval-first', default=False, action='store_true',
                    help='evaluate model weights before training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--backup-iters', type=int, default=1000,
                    help='iteration interval to perform state backups')
parser.add_argument('--batch-size', default=3, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--patience', default=2, type=int,
                    help='patience epochs for LR decreasing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--iou-loss', action='store_true',
                    help='use IoULoss instead of BCE')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='epoch number to resume')
parser.add_argument('--optim-snapshot', type=str,
                    default='./weights/LiverNet_train_optim.pth',
                    help='path to optimizer state snapshot')
parser.add_argument('--accum-iters', default=1, type=int,
                     help='number of gradient accumulated iterations to wait '
                          'before update')
parser.add_argument('--pin-memory', default=False, action='store_true',
                     help='enable CUDA memory pin on DataLoader')

# Model settings
parser.add_argument('--size', default=512, type=int,
                    help='image size')

# Other settings
parser.add_argument('--visdom', type=str, default=None,
                    help='visdom URL endpoint')
parser.add_argument('--env', type=str, default='LiverNet-train',
                    help='visdom environment name')

args = parser.parse_args()

args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
print('\n\n')

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

image_size = (args.size, args.size)

input_transform = Compose([
    # ResizeImage(args.size),
    # ToTensor(),    
    Normalize(
        mean=[-50164797.52292029],
        std=[238999.29191403434])
])

# If we are in 'low res' mode, downsample the target
# target_transform = Compose([
#     # ToTensor(),
#     ResizeAnnotation(args.size),
# ])

# if args.high_res:
#     target_transform = Compose([
#         # ToTensor()
#     ])

if args.batch_size == 1:
    args.time = -1

liver = LiverDataset(data_root=args.data, split=args.split, 
                     data_folder=args.split_root, transform=input_transform)
                     # annotation_transform=annotation_transform)

train_loader = DataLoader(liver, batch_size=args.batch_size, shuffle=True,
                          pin_memory=args.pin_memory, num_workers=args.workers)

start_epoch = args.start_epoch

if args.val is not None:
    val_liver = LiverDataset(data_root=args.data, split=args.val, 
                     data_folder=args.split_root, transform=input_transform)
                     # annotation_transform=annotation_transform)
    val_loader = DataLoader(val_liver, batch_size=args.batch_size, shuffle=False,
                          pin_memory=args.pin_memory, num_workers=args.workers)



if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

net = PSPNet(n_classes = 3, psp_size=512)

if osp.exists(args.snapshot):
    print('Loading state dict from: {0}'.format(args.snapshot))
    snapshot_dict = torch.load(args.snapshot)
    net.load_state_dict(snapshot_dict)

if args.cuda:
    net.cuda()

if args.visdom is not None:
    visdom_url = urlparse(args.visdom)

    port = 80
    if visdom_url.port is not None:
        port = visdom_url.port

    print('Initializing Visdom frontend at: {0}:{1}'.format(
          args.visdom, port))
    vis = VisdomWrapper(server=visdom_url.geturl(), port=port, env=args.env)

    if args.val is not None:
        vis.init_line_plot('val_plt', xlabel='Epoch', ylabel='IoU',
                           title='Current Model IoU Value',
                           legend=['Loss'])

optimizer = optim.Adam(net.parameters(), lr=args.lr)

scheduler = ReduceLROnPlateau(
    optimizer, patience=args.patience)

if osp.exists(args.optim_snapshot):
    optimizer.load_state_dict(torch.load(args.optim_snapshot))
    # last_epoch = args.start_epoch

scheduler.step(args.start_epoch)

criterion = nn.CrossEntropyLoss()

def train(epoch):
    net.train()
    total_loss = AverageMeter()
    # total_loss = 0
    epoch_loss_stats = AverageMeter()
    time_stats = AverageMeter()
    # epoch_total_loss = 0
    start_time = time.time()
    optimizer.zero_grad()
    loss = 0
    # pdb.set_trace()
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        # imgs = Variable(imgs)
        imgs = imgs.requires_grad_()
        # masks = Variable(masks) # masks.squeeze())
        masks = masks.requires_grad_()

        if args.cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()

        out_masks = net(imgs)
        loss += criterion(out_masks, masks)

        
        if batch_idx % args.accum_iters == 0:
            loss = loss / args.accum_iters
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item(), imgs.size(0))
            epoch_loss_stats.update(loss.item(), imgs.size(0))
            time_stats.update(time.time() - start_time)
            # start_time = time.time()
            loss = 0
            optimizer.zero_grad()
        # total_loss += loss.data[0]
        # epoch_total_loss += total_loss

        if args.visdom is not None:
            cur_iter = batch_idx + (epoch - 1) * len(train_loader)
            vis.plot_line('iteration_plt',
                          X=torch.ones((1, 1)).cpu() * cur_iter,
                          Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                          update='append')

        if batch_idx % args.backup_iters == 0:
            filename = 'LiverNet_{0}_snapshot.pth'.format(args.split)
            filename = osp.join(args.save_folder, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = 'LiverNet_{0}_optim.pth'.format(args.split)
            optim_filename = osp.join(args.save_folder, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if batch_idx % args.log_interval == 0:
            # elapsed_time = time.time() - start_time
            elapsed_time = time_stats.avg
            # cur_loss = total_loss / args.log_interval
            print('[{:5d}] ({:5d}/{:5d}) | ms/batch {:.6f} |'
                  ' loss {:.6f} | lr {:.7f}'.format(
                      epoch, batch_idx, len(train_loader),
                      elapsed_time * 1000, total_loss.avg,
                      optimizer.param_groups[0]['lr']))
            total_loss.reset()

        # total_loss = 0
        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg

    if args.visdom is not None:
        vis.plot_line('epoch_plt',
                      X=torch.ones((1, 1)).cpu() * epoch,
                      Y=torch.Tensor([epoch_total_loss]).unsqueeze(0).cpu(),
                      update='append')
    return epoch_total_loss


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum() + 1e-9
    return intersection / masks.size(0), union / masks.size(0)


def evaluate(epoch=0):
    net.train()
    score_thresh = np.concatenate([# [0],
                                   # np.logspace(start=-16, stop=-2, num=10,
                                   #             endpoint=True),
                                   np.arange(start=0.00, stop=0.96,
                                             step=0.025)]).tolist()
    cum_I = torch.zeros(3, len(score_thresh))
    cum_U = torch.zeros(3, len(score_thresh))
    eval_seg_iou_list = [.5, .6, .7, .8, .9]

    seg_correct = torch.zeros(3, len(eval_seg_iou_list), len(score_thresh))
    seg_total = 0
    start_time = time.time()
    for img, mask in tqdm(val_loader, dynamic_ncols=True):
        # imgs = Variable(img, volatile=True)
        # imgs = img.requires_grad_()
        # mask = mask.squeeze()
        # mask = mask.requires_grad_()

        if args.cuda:
            imgs = img.cuda()
            mask = mask.float().cuda()
            # print('-----------mask------------')
            # print(mask.shape)

        with torch.no_grad():
            out = net(imgs) #, words)
            out = F.softmax(out, dim=1)
        # pdb.set_trace()
        for cat in range(0, 3):
            out_mask = out[:, cat]
            cat_mask = (mask == cat).float()
            inter = torch.zeros(len(score_thresh))
            union = torch.zeros(len(score_thresh))
            for idx, thresh in enumerate(score_thresh):
                thresholded_out = (out_mask > thresh).float().data
                # print(thresholded_out.shape)
                # print('*************************')
                try:
                    inter[idx], union[idx] = compute_mask_IU(thresholded_out, cat_mask)
                except AssertionError as e:
                    inter[idx] = 0
                    union[idx] = cat_mask.sum()

            cum_I[cat] += inter
            cum_U[cat] += union
            this_iou = inter / (union + 1e-9)

            for idx, seg_iou in enumerate(eval_seg_iou_list):
                for jdx in range(len(score_thresh)):
                    seg_correct[cat, idx, jdx] += (this_iou[jdx] >= seg_iou).float()
                    
        seg_total += imgs.size(0)

    cat_iou = {}
    for cat in range(0, 3):
        # Evaluation finished. Compute total IoUs and threshold that maximizes
        print("Category {0}".format(cat))
        for jdx, thresh in enumerate(score_thresh):
            print('-' * 32)
            print('precision@X for Threshold {:<15.3E}'.format(thresh))
            for idx, seg_iou in enumerate(eval_seg_iou_list):
                print('precision@{:s} = {:.5f}'.format(
                    str(seg_iou), seg_correct[cat, idx, jdx] / seg_total))

        # Print final accumulated IoUs
        final_ious = cum_I[cat] / (cum_U[cat] + 1e-9)
        print('-' * 32 + '\n' + '')
        print('FINAL accumulated IoUs at different thresholds:')
        print('{:15}| {:15} |'.format('Thresholds', 'mIoU'))
        print('-' * 32)
        for idx, thresh in enumerate(score_thresh):
            print('{:<15.3E}| {:<15.13f} |'.format(thresh, final_ious[idx]))
        print('-' * 32)

        max_iou, max_idx = torch.max(final_ious, 0)
        max_iou = float(max_iou.numpy())
        max_idx = int(max_idx.numpy())

        # Print maximum IoU
        print('Evaluation done. Elapsed time: {:.3f} (s) '.format(
            time.time() - start_time))
        print('Maximum IoU: {:<15.13f} - Threshold: {:<15.13f}'.format(
            max_iou, score_thresh[max_idx]))
        cat_iou[cat] = max_iou

    return cat_iou

if __name__ == '__main__':
    print('Train begins...')
    best_val_loss = None
    if args.eval_first:
        evaluate(0)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            #if epoch == 1:
            #    train_loss = torch.load('./weights/train_loss.pth')
            #else:
            train_loss = train(epoch)
            file_na = osp.join(args.save_folder, 'train_loss.pth')
            torch.save(train_loss, file_na)
            val_loss = train_loss
            if args.val is not None:
                cat_iou = evaluate(epoch)
                bg_iou = cat_iou[0]
                liver_iou = cat_iou[1]
                tumor_iou = cat_iou[2]
                val_loss = 1 - np.mean([bg_iou, liver_iou, tumor_iou])
                flena = osp.join(args.save_folder, 'val_loss.pth')
                torch.save(val_loss, flena)
            scheduler.step(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| epoch loss {:.6f} |'.format(
                      epoch, time.time() - epoch_start_time, train_loss))
            print('-' * 89)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
            filename = osp.join(args.save_folder, 'LiverNet_best_weights.pth')
            torch.save(net.state_dict(), filename)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


