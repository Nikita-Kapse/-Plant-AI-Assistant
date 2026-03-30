# from ultralytics import YOLO

# # Load the YOLOv8 stress classifier model
# stress_model = YOLO("models/stress_classifier.pt")

# def classify_stress(image_path):
#     """
#     Classify a leaf image into stress categories using YOLOv8.

#     Args:
#         image_path (str): Path to the leaf image.

#     Returns:
#         tuple: (predicted_class_label, confidence_score)
#                Class labels: "biotic", "abiotic", or "healthy"
#     """
#     results = stress_model(image_path)

#     class_id = results[0].probs.top1
#     confidence = float(results[0].probs.top1conf)
#     class_label = stress_model.names[class_id]

#     return class_label, confidence


# if __name__ == "__main__":
#     image_path = "leaf.jpeg"
#     label, conf = classify_stress(image_path)
#     print(f"Stress Type: {label}")
#     print(f"Confidence: {conf:.4f}")


from ultralytics import YOLO

model = YOLO("models/stress_classifier.pt")

def classify_stress(image_path):

    results = model(image_path)

    probs = results[0].probs

    class_id = probs.top1
    confidence = float(probs.top1conf)

    label = model.names[class_id]

    return label, confidence