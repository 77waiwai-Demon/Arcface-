import torch
from torch import nn
from torch.utils import data as Data
from glob import glob
import re
import random
import cv2
from config import Config
import numpy as np
from torch.optim.lr_scheduler import StepLR
from models import resnet_face18, ArcMarginProduct, FocalLoss, cosin_metric
from tqdm import tqdm
import math
import os

config = Config.from_json_file("config.json")
is_cuda = torch.cuda.is_available()
print(is_cuda)


# 数据读取
def read_data(train_test_split=0.90):

    path = glob("./face_data/*")
    new_path = []
    label_dict = dict()
    for img_path in path:
        label = img_path.split("\\")[-1]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        p = glob(img_path+"/*")
        if len(p) > 50:
            new_path.extend(random.sample(p, 50))
        else:
            new_path.extend(p)
    # 数据分集
    random.shuffle(new_path)
    print(len(new_path))
    train_idx = int(len(new_path) * train_test_split)

    train_path = new_path[:train_idx]
    test_path = new_path[train_idx:]


    return train_path, test_path, label_dict

# 自定义数据集
class FaceData(Data.Dataset):
    def __init__(self, path, label_dict):
        self.path = path
        self.label_dict = label_dict

    def to_tuple(self, x):
        return x, x

    def read_img(self, path):

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.to_tuple(config.image_size)) / 255.0
        img = img[None, :, :]
        # img = np.transpose(img, (2, 0, 1))
        return np.float32(img)

    def __getitem__(self, item):
        path = self.path[item]
        value = self.read_img(path)
        label_str = path.split("\\")[-2]
        label = self.label_dict[label_str]
        return value, np.int64(label)

    def __len__(self):
        return len(self.path)

class StepLRS(StepLR):

    def __init__(self, optimizer, step_size, last_lr, k, b, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.last_lr = last_lr
        self.k = k
        self.b = b
        super(StepLRS, self).__init__(optimizer, step_size, gamma)

    def get_lr(self):

        lr = [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        # lr = [self.k * self.last_epoch + self.b for _ in self.optimizer.param_groups]
        if lr[0] < self.last_lr:
            return [group['lr'] for group in self.optimizer.param_groups]

        # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        #     return [group['lr'] for group in self.optimizer.param_groups]
        return lr

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


def train():
    train_path, test_path, label_dict = read_data()

    train_data = FaceData(train_path, label_dict)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    test_data = FaceData(test_path, label_dict)
    test_data = Data.DataLoader(test_data, shuffle=True, batch_size=config.batch_size)

    all_step = len(train_data) * config.epochs

    # 初始化网络
    model = resnet_face18()
    arc_model = ArcMarginProduct(512, len(label_dict))


    if os.path.exists("arcFaceModel.pkl"):
        model.load_state_dict(torch.load("arcFaceModel.pkl"))

    model.train()
    arc_model.train()

    if is_cuda:
        model.cuda()
        arc_model.cuda()

    k = (config.lr - config.last_lr) / -(all_step * 0.9)
    b = config.lr


    # 初始化参数
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': arc_model.parameters()}], lr=config.lr, weight_decay=5e-4)
    gamma = math.pow(config.last_lr / config.lr, 1 / (all_step * 0.9))
    sc_optimizer = StepLRS(optimizer, 1, config.last_lr, gamma=gamma, k=k, b=b)
    loss_fc = FocalLoss(1.0)

    old_loss = 100
    nb = len(train_data)
    for epoch in range(1, config.epochs+1):
        pbar = tqdm(train_data, total=nb)
        loss_all = 0
        acc_all = 0
        for step, (x, y) in enumerate(pbar):

            if is_cuda:
                x, y = x.cuda(), y.cuda()
            out = model(x)

            out_class = arc_model(out, y)
            loss = loss_fc(out_class, y)

            loss.backward()
            optimizer.step()
            sc_optimizer.step()
            optimizer.zero_grad()

            lr = optimizer.param_groups[0]["lr"]
            acc = torch.mean((torch.argmax(out_class, dim=1) == y).float())

            loss_all += loss.item()
            loss_time = loss_all / (step+1)

            acc_all += acc
            acc_time = acc_all / (step+1)

            s = ("train => epoch:{} - step:{} - loss:{:.3f} - loss_time:{:.3f} - acc:{:.3f} - acc_time:{:.3f} - lr:{:.6f}".format(epoch, step, loss, loss_time, acc, acc_time, lr))
            pbar.set_description(s)

        with torch.no_grad():
            model.eval()
            arc_model.eval()
            test_pbar = tqdm(test_data)
            loss_all = 0
            acc_all = 0
            for step, (x, y) in enumerate(test_pbar):
                if is_cuda:
                    x, y = x.cuda(), y.cuda()
                out = model(x)
                out_class = arc_model(out, y)
                loss = loss_fc(out_class, y)
                acc = torch.mean((torch.argmax(out_class, dim=1) == y).float())

                loss_all += loss.item()
                loss_time = loss_all / (step + 1)

                acc_all += acc
                acc_time = acc_all / (step + 1)

                s = (
                    "test => epoch:{} - step:{} - loss:{:.3f} - loss_time:{:.3f} - acc:{:.3f} - acc_time:{:.3f}".format(
                        epoch, step, loss, loss_time, acc, acc_time))
                test_pbar.set_description(s)

        model.train()
        arc_model.train()

        if old_loss > loss_time:
            old_loss = loss_time
            torch.save(model.state_dict(), "arcFaceModel.pkl")
            # torch.save(arc_model.state_dict(), "arcModel1.pkl")


if __name__ == '__main__':
    train()

