from ultralytics import YOLO
model = YOLO("models/best.pt")

print("Model loaded successfully")
print(model.names)