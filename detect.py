import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp2/weights/best.pt') # select your model.pt path
    model.predict(
                  # source = 'E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/video/output_video10.avi',
        source='images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )