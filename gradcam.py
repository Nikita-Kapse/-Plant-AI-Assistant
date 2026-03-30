import torch
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")
model.model.eval()

features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer = model.model.model[-3]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam(image_path):

    image = cv2.imread(image_path)

    original = image.copy()

    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image).unsqueeze(0)
    image.requires_grad = True

    logits = model.model(image)[0]

    class_idx = torch.argmax(logits)

    score = logits[0, class_idx]

    model.model.zero_grad()
    score.backward()
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    cam = torch.sum(weights * features, dim=1)
    cam = cam.squeeze().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay


if __name__ == "__main__":

    image_path = "leaf.jpeg"
    heatmap = generate_gradcam(image_path)
    cv2.imwrite("heatmap_result.jpg", heatmap)

    print("Heatmap saved as heatmap_result.jpg")