from ultralytics import YOLO

def load_model():
    model = YOLO("models/best.pt")
    return model

def predict_image(model, image_path):

    results = model(image_path)
    result = results[0]

    class_id = result.probs.top1
    confidence = float(result.probs.top1conf)
    class_name = model.names[class_id]

    return class_name, confidence

if __name__ == "__main__":

    model = load_model()
    image_path = "leaf.jpeg"
    disease, confidence = predict_image(model, image_path)

    print("Predicted Disease:", disease)
    print("Confidence:", confidence)