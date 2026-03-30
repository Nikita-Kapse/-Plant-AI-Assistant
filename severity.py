import cv2
import numpy as np

def compute_severity(heatmap_path):
    heatmap = cv2.imread(heatmap_path)

    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    infected_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    severity = (infected_pixels / total_pixels) * 100

    return severity

if __name__ == "__main__":

    severity = compute_severity("heatmap_result.jpg")
    print("Disease Severity:", round(severity, 2), "%")