import argparse
import time
from pathlib import Path

import os
from os.path import join as pjoin

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from Yolov5_Web.models.experimental import attempt_load
from Yolov5_Web.utils.datasets import LoadStreams, LoadImages
from Yolov5_Web.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from Yolov5_Web.utils.plots import plot_one_box
from Yolov5_Web.utils.torch_utils import select_device, load_classifier, time_synchronized


def logging(s, log_path='runs/error/log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+', encoding='utf-8') as f_log:
            f_log.write(s + '\n')

def detect():
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size

    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    # 处理数据集文件夹
    for i in os.listdir(source):
        source0 = pjoin(source,i)  #得到子文件夹路径
        filename = os.path.basename(source0)  # 获取文件夹名
        if len(os.listdir(source0)) == 0:
            logging('没有图片:{}'.format(filename))
            continue
        webcam = source0.isnumeric() or source0.endswith('.txt') or source0.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

        # 目录配置
        save_dir = Path(increment_path(Path(opt.project) / filename, exist_ok=opt.exist_ok))  # 增量运行
        (save_dir if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16


        # 数据加载
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source0, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source0, img_size=imgsz, stride=stride)


        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                txt_path = str(save_dir / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            line = (cls, *xyxy)  # 保存目标的类别与四个坐标
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

        if save_txt:
            print(f"{len(list(save_dir.glob('*.txt')))} labels saved to {save_dir}" if save_txt else '')
        print('Done. (%.3fs) \n' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/DataSet/new_bank_data/train_data/lip', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all Lip_models')
    parser.add_argument('--project', default='D:/DataSet/new_bank_data/train_root', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all Lip_models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


