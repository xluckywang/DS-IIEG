from ultralytics import YOLO
import wandb

# Load a model
# model = YOLO('yolov8s.yaml')  # build a new model from YAML
model = YOLO(r'./yolov8n-cls.pt')
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transf er weights

# Train the model
if __name__ == '__main__':
    wandb.login(key="9b8156cda21fc9551c09f98a60e7c4b3dd64b378")
    # model.train(data=r'./../data/data.yaml', epochs=500, imgsz=640,device='cuda')  # device指定0号GPU执行
    model.train(data=r'../../0_data/datasetOri/MTD', epochs=500, imgsz=224, device='cuda',patience=100,cache=False)  # device指定0号GPU执行
    # model.train(data=r'../data/datasetA/MTD_train_multi', epochs=500, imgsz=640, device='cuda',cache=False,patience=200)
    model.eval()