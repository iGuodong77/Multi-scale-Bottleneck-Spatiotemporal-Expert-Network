import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
from datasets.CDdataset import load_and_split, hyper_CDDataset
import os
import random
import cv2 as cv
import time
from tqdm import tqdm
import numpy as np
import argparse
from network.baseline import CD_Model_diff
from multi_similarity_loss import MultiSimilarityLoss
from network.loss import FocalLoss, DiceLoss, CombinedLoss
from datetime import datetime
from accuracy import accuracy_indicators as acc
import re

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_name', type=str, default='quyu3')  # river    farm    Hermiston    bayArea    santaBarbara
parser.add_argument('--data_path', type=str, default='/home/HDD/林地变化检测/Data/林地一期')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=200)
# parser.add_argument('--frft_order', type=float, default=[0.1, 0.5, 0.9])  # [0.1, 0.5, 0.9]
parser.add_argument('--gpu', type=int, default=0, help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--patch_size', type=int, default=7, help="Size of the spatial neighbourhood")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate, set by the model if not specified.")
parser.add_argument('--arg_1', type=float, default=1)
# parser.add_argument('--train_ratio', type=float, default=0.05, help="train_ratio")
parser.add_argument('--l2_decay', type=float, default=5e-4, help='the L2  weight decay')
# parser.add_argument('--margin', type=float, default=0.25, help="Metrix margin")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--num_worker', type=int, default=10, help="num_worker")
args = parser.parse_args()


def seed_worker(seed):
    # 固定随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(net, val_loader):
    pre = []
    ys = []
    for i, batch in enumerate(val_loader):
        with torch.no_grad():
            xA = batch['data'][0].to(args.gpu)
            xB = batch['data'][1].to(args.gpu)
            y = batch['label']
            cls = net(xA, xB)
            cls = cls.argmax(dim=1)
            pre.append(cls.detach().cpu().numpy())
            ys.append(y.numpy())
    pre = np.concatenate(pre)
    ys = np.concatenate(ys)
    accuracy = np.mean(ys == pre) * 100
    return accuracy


def inference(net, labels, test_loader, log_dir):
    net.init_weight()
    saved_weight = torch.load(os.path.join(log_dir, 'best.pkl'))
    net.load_state_dict(saved_weight['Discriminator'])
    net.eval()
    out = []
    t1 = time.time()
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for batch in tbar:
            xA = batch['data'][0].to(args.gpu)
            xB = batch['data'][1].to(args.gpu)
            cls = net(xA, xB)
            cd_preds = cls.argmax(dim=1)
            out.append(cd_preds.detach().cpu().numpy())
    pre = np.concatenate(out)
    pre = pre.reshape(labels.shape[0], labels.shape[1])
    if args.data_name == 'quyu1' or args.data_name == 'quyu2' or args.data_name == "quyu3" or args.data_name == 'quyu4' or args.data_name == "quyu5":
        pre = pre + 1
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 0:
                    pre[i, j] = 0
                else:
                    continue
    oa, kappa_co, F1, Pr, Re, conf_mat = acc(pre, labels, data_name=args.data_name)
    t2 = time.time()
    result = f"OA: {oa:.4f}    Kappa: {kappa_co:.4f}    F1: {F1:.4f}    Pr: {Pr:.4f}    Re: {Re:.4f}    conf_mat:"
    print(result, conf_mat)

    test_time = f'test is done! test time {t2-t1:.2f}'
    print(test_time)

    with open(os.path.join(log_dir, 'result.txt'), 'a+') as fout:
        fout.write(result + str(conf_mat) + '\n')
        fout.write(str(test_time) + '\n')
    if args.data_name == 'quyu1' or args.data_name == "quyu2" or args.data_name == "quyu3" or args.data_name == 'quyu4' or args.data_name == "quyu5":
        # pre = pre / 2
        pre = np.where(pre == 0, 1, pre)
        pre = pre - 1

    pre = pre * 255
    # pre = np.expand_dims(pre, axis=-1)
    # pre = np.concatenate([pre, pre, pre], axis=2)
    cv.imwrite(os.path.join(log_dir, "output.png"), pre)


def experiment():
    # 创建训练文件夹
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    match = re.search(r'(林地一期|林地二期)', args.data_path)
    project_name = match.group(1) if match else 'None_Project'
    root = os.path.join(args.save_path, time_str, project_name)
    log_dir = os.path.join(root, args.data_name)

    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'train_log.txt'), 'a+') as fout:
        fout.write(str(hyperparams) + '\n')

    seed_worker(args.seed)

    # 读数据并按标签划分训练集、验证集、测试集
    coord_label = load_and_split(args)
    im1, im2, labels, N_BANDS = coord_label.load_data()
    train_coord, train_label, test_coord = coord_label.cood_split()
    hyperparams.update({'n_bands': N_BANDS, 'device': args.gpu})

    # 创建模型、优化器、学习率调整器
    net = CD_Model_diff(inchannel=N_BANDS, patch_size=args.patch_size).to(args.gpu)
    # opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2_decay)
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, [100, 130, 160], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.max_epoch // 10, gamma=0.8)
    # scheduler = optim.lr_scheduler.MultiStepLR(opt, [170, 180, 190], gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

    # 构建损失 - 使用组合损失提升Kappa和F1
    # cls_criterion = nn.CrossEntropyLoss()
    # cls_criterion = FocalLoss(alpha=0.75, gamma=2.0)
    cls_criterion = CombinedLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)  # Focal + Dice
    # ms_criterion = MultiSimilarityLoss(margin=args.margin).to(args.gpu)

    # 开始训练
    best_acc = 0
    taracc, taracc_list = 0, []
    early_patience = 0
    lr_patience = 0
    for epoch in range(1, args.max_epoch+1):
        
        # 导入训练集
        train_data = hyper_CDDataset(args, train_coord, im1, im2, train_label)
        train_loader = Data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True)

        t1 = time.time()
        loss_list = []
        cls_correct = 0
        label_num = 0
        net.train()
        # 训练阶段
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
            xA = batch['data'][0].to(args.gpu)
            xB = batch['data'][1].to(args.gpu)
            y = batch['label'].to(args.gpu)
            opt.zero_grad()

            # cls = net(xA, xB)
            cls = net(xA, xB)
            loss_cls = cls_criterion(cls, y.long())
            # con_loss = supervised_contrastive_loss(proj, y, temperature=0.1)
            # loss_ms = ms_criterion(proj, y)

            loss = loss_cls
            # loss = loss_cls

            loss.backward()
            opt.step()
            #
            loss_list.append([loss_cls.item(), loss.item()])
            # loss_list.append([loss_cls.item(), loss_ms, loss.item()])
            cls = cls.detach().argmax(dim=1, keepdim=True)

            cls_correct += cls.eq(y.view_as(cls)).sum().item()
            label_num += len(y)
        train_acc = 100 * cls_correct / label_num
        cls_loss, total_loss = np.mean(loss_list, 0)
        t2 = time.time()
        # print('epoch {}:  Order_1 = {} Order_2 = {} Order_3 = {}'.format(epoch, net.order[0].data, net.order[1].data, net.order[2].data))

        lr_info = opt.param_groups[0]["lr"]
        scheduler.step()
        # 打印信息
        train_info = f'epoch {epoch}, train time {t2 - t1:.2f}, lr {lr_info:.0e} cls_loss {cls_loss:.4f} ' \
                         f'total_loss {total_loss:.4f} train_acc {train_acc:2.2f}'
        print(train_info)
        with open(os.path.join(log_dir, 'train_log.txt'), 'a+') as fout:
            fout.write(str(train_info) + '\n')
    # 保存模型
    torch.save({'Discriminator': net.state_dict()}, os.path.join(log_dir, f'best.pkl'))
    print('train is done!')

    # 测试阶段， 导入测试集
    test_data = hyper_CDDataset(args, test_coord, im1, im2)
    test_loader = Data.DataLoader(test_data, batch_size=2048, num_workers=args.num_worker, shuffle=False)
    # log_dir = '/home/HDD/林地变化检测/demo-forest/results/quyu5/11-14_15-53-45'
    taracc = inference(net, labels, test_loader, log_dir)


if __name__ == '__main__':
    experiment()

