import cv2
import torch
from glob import glob
from mtcnn import FaceDetector
import math
from PIL import Image
import numpy as np
from models import resnet_face18, cosin_metric
from config import Config
import os
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化网络
detector = FaceDetector()
config = Config.from_json_file("config.json")
model = resnet_face18()

model.load_state_dict(torch.load("arcFaceModel.pkl", map_location=torch.device(device)))
model.eval()
model.to(device)

names = []
pa = glob("./face_database/*")
for p in pa:
    if not names:
        n = re.findall("\\\\(.+)\.n", p)[0]
        names.append(n)

lens = len(names)

sce = config.s # 阈值

# 图片转换
def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 图片转换
def pil_to_cv2(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 人脸矫正
def face_correct(img):
    # bbox：[左上角x坐标, 左上角y坐标, 右下角x坐标, 右下角y坐标, 检测评分]
    # landmark：[右眼x, 左眼x, 鼻子x, 右嘴角x, 左嘴角x, 右眼y, 左眼y, 鼻子y, 右嘴角y, 左嘴角y]
    bboxes, landmarks = detector.detect(img)

    # 取出框最大的脸
    faces = []
    for box, land in zip(bboxes, landmarks):
        left_eye = [land[1], land[6]]
        right_eye = [land[0], land[5]]
        k = (left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0])
        arc = math.atan(k)
        angle = arc * 180 / math.pi
        img = img.rotate(angle)
        face_img = img.crop((box[0],box[1], box[2], box[3]))
        faces.append([face_img, face_img.size[0]*face_img.size[1]])

    faces.sort(key= lambda x:x[1], reverse=True)
    face_img = faces[0][0]
    return face_img

# 保存人脸
def save_face(img):
    img = cv2_to_pil(img)
    face_img = face_correct(img)
    face_img = pil_to_cv2(face_img)
    face_img = pre_img(face_img)
    name = str(input("人脸写入成功，请输入你的名字："))
    np.save(f"./face_database/{name}", face_img)
    print("录入成功")

def open_cap():
    # 开启摄像头
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"开启了 {i}号 摄像头")
            break
    else:
        print("摄像头开启失败，请检查摄像头，回车程序结束")
        input("")
        exit()

    return cap

def read_face():
    paths = glob("./face_database/*")
    images = []

    for p in paths:
        img = np.load(p)
        images.append(img)


    x = np.concatenate(images, axis=0)
    return x

def pre_img(img):
    img = cv2.resize(img, (config.image_size,) * 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img[None, None, :, :] / 127.5 - 1.0)
    return img

def read_img(cap):
    cap = open_cap()
    ret, img = cap.read()
    if not ret:
        assert "摄像头图片读取失败"

    return img

def main():
    # 程序开启
    while True:
        print("请输入模式：")

        print("1.签到")
        print("2.录入人脸")
        print("3.人脸库识别")

        kind = int(input("模式选择(输入数字123，回车结束)："))

        if kind == 1:
            if not names:
                print("已经全部签到")
                continue
            # 检查人脸库是否有人
            if not glob("./face_database/*"):
                print("抱歉，人脸库中未存在人脸数据，请先录入人脸")
                continue

            images = read_face()
            cap = open_cap()
            img = read_img(cap)
            img = cv2_to_pil(img)
            img = face_correct(img)
            img = pil_to_cv2(img)
            img = pre_img(img)

            x = np.concatenate([img, images], axis=0)
            x = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                pred = model(x)

            you = pred[-1].cpu().numpy()
            other = pred[:-1].cpu().numpy()
            lens = other.shape[0]

            cs = []
            scores = []
            for i in range(lens):
                cos_sore = cosin_metric(you, other[i])
                cs.append(cos_sore>sce)
                scores.append(cos_sore)
            if True in cs:
                index = cs.index(True)
                name = names.pop(index)
                l = len(names)
                
                print(f"相似度：{scores[index]}")
                print(f"{name} 签到成功")

                print(f"一共{lens}人，已签到 {lens-l} 人，未签到 {l} 人")

                if l > 0:
                    w_name = "  ".join(names)
                    print(f"未签到人名单：{w_name}")

            else:
                print(scores)
                print("识别失败，你不是本公司人员")

            cap.release()

            continue

        elif kind == 2:
            cap = open_cap()
            img = read_img(cap)
            cap.release()
            save_face(img)
            continue

        elif kind == 3:
            # 检查人脸库是否有人
            if not glob("./face_database/*"):
                print("抱歉，人脸库中未存在人脸数据，请先录入人脸")
                continue

            images = read_face()
            cap = open_cap()
            img = read_img(cap)
            img = cv2_to_pil(img)
            img = face_correct(img)
            img = pil_to_cv2(img)
            img = pre_img(img)

            x = np.concatenate([img, images], axis=0)
            x = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                pred = model(x)

            you = pred[-1].cpu().numpy()
            other = pred[:-1].cpu().numpy()
            lens = other.shape[0]

            cs = []
            scores = []
            for i in range(lens):
                cos_sore = cosin_metric(you, other[i])
                cs.append(cos_sore > sce)
                scores.append(cos_sore)
            if True in cs:
                index = cs.index(True)
                print(f"相似度：{scores[index]}")
                print("识别成功")
            else:
                print(scores)
                print("识别失败")

            cap.release()

            continue

if __name__ == '__main__':

    main()