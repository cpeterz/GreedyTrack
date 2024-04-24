import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-contex-cpms.yaml')
    model.load('yolov8s.pt') # loading pretrain weights
    model.train(data='E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/dataset/uavchange/uavchange.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/runs/train/exp3/weights/last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )