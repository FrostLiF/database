from ultralytics import YOLO

model = YOLO("yolov11n.pt")
model.train(
    data="datasets/anime_faces/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="anime_facial_parts",
    exist_ok=True
)
