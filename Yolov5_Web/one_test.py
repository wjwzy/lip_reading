import torch
import torch.backends.cudnn as cudnn
import numpy as np
from Yolov5_Web.models.experimental import attempt_load
from Yolov5_Web.utils.datasets import letterbox
from Yolov5_Web.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from Yolov5_Web.utils.torch_utils import select_device

class yolov5(object):
    def __init__(self,
                 img_size = 416,
                 weights = 'runs/train/exp/weights/best.pt',
                 iou_thres = 0.45,
                 conf_thres = 0.25,
                 device = '0',
                 classes = 0,
                 agnostic_nms = False,
                 augment = False
                 ):
        self.imgsz = img_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.device = select_device(device)
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        # Initialize
        set_logging()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

    def detect(self, source):
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # 检查图片的大小
        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True  # 设置True可以加速恒定图像大小的处理速度

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        result = []
        for img0 in source:
            imgsz = check_img_size(imgsz)  # check img_size
            img = letterbox(img0, imgsz, stride=32)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 获取模型预测
            pred = self.model(img, augment=self.augment)[0]

            # 使用NMS进行预测
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)

            # 过程检测
            for i, det in enumerate(pred):  # 遍历预测框
                # 还原图像坐标值大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                result.append(det[0][:4].tolist())

        return result


if __name__ == '__main__':
    yolov5().detect('data/images/')
