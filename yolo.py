from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('input_vids/08fd33_4.mp4', save=True)
print(results[0])
print("====", '\n')

for box in results[0].boxes:
    print(box)



