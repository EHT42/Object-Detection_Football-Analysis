from ultralytics import YOLO
import torch


yaml_data_path = r'C:\Users\yunge\PycharmProjects\ML\new_football_analysis\training\football-players-detection-1\data.yaml'


def gpu_check():
    print(torch.cuda.is_available())  # True, wenn die GPU verfügbar ist
    print(torch.cuda.device_count())  # Gibt die Anzahl der GPUs aus
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))

    print("================================")


def run_model(data_path):
    model = YOLO('yolov8x.pt')
    path = data_path
    model.train(data=data_path, epochs=100, imgsz=640)


if __name__ == '__main__':
    check_gpu = gpu_check()
    training = run_model(yaml_data_path)