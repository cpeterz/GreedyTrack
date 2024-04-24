import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/runs/train/exp3/weights/best.pt') # select your model.pt path
    model.track(source='E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/video/output_video10_3.avi',
                tracker='E:/YOLOV8改进策略/全包/面具/ultralytics-20240222/ultralytics-main/ultralytics/cfg/trackers/bytetrack.yaml',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True,
                )