from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
 
if __name__ == '__main__':
 
 
    # 加载模型
    model = YOLO("ultralytics/cfg/models/11/yolo11_LDConv.yaml")  # 你要选择的模型yaml文件地址
    # Use the model
    results = model.train(data=r"C:\\Users\\User\\YOLO11\\ultralytics\\dataset\\550andcustomthermal\\data.yaml",
                          epochs=60, batch=16, imgsz=640, workers=4, name=Path(model.cfg).stem)  # 训练模型