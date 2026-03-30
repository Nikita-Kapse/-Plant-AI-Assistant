import gradio as gr
import cv2
import os

from predict import load_model, predict_image
from gradcam import generate_gradcam
from severity import compute_severity
from stress_classifier import classify_stress
from abiotic_classifier import classify_abiotic
from yield_model import predict_yield   # ✅ NEW IMPORT

# Load model once
disease_model = load_model()


def analyze_leaf(image):

    if image is None:
        return None, None, "Please upload an image."

    try:
        # Save image safely
        image_path = "temp_image.jpg"
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # ── STEP 1: Stress Classification ──
        stress_type, stress_conf = classify_stress(image_path)

        # Normalize label
        stress_type = stress_type.strip().lower()

        print(f"[DEBUG] Stress: {stress_type} | Conf: {stress_conf:.4f}")

        # ───────────── BIOTIC ─────────────
        if stress_type == "biotic":

            disease, disease_conf = predict_image(disease_model, image_path)

            heatmap = generate_gradcam(image_path)

            if heatmap is not None:
                cv2.imwrite("heatmap.jpg", heatmap)
                severity = compute_severity("heatmap.jpg")
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            else:
                severity = 0
                heatmap_rgb = None

            result = (
                f"🔬 Stress Type: Biotic\n"
                f"Confidence: {stress_conf:.3f}\n\n"
                f"🦠 Disease: {disease}\n"
                f"Confidence: {disease_conf:.3f}\n\n"
                f"📊 Severity: {severity:.2f}%"
            )

            return image, heatmap_rgb, result

        # ───────────── ABIOTIC ─────────────
        elif stress_type == "abiotic":

            abiotic_type, abiotic_conf = classify_abiotic(image_path)

            result = (
                f"🔬 Stress Type: Abiotic\n"
                f"Confidence: {stress_conf:.3f}\n\n"
                f"🌡️ Type: {abiotic_type}\n"
                f"Confidence: {abiotic_conf:.3f}"
            )

            return image, None, result

        # ───────────── HEALTHY ─────────────
        else:

            result = (
                f"🔬 Stress Type: Healthy\n"
                f"Confidence: {stress_conf:.3f}\n\n"
                f"✅ Leaf appears healthy."
            )

            return image, None, result

    except Exception as e:
        print("[ERROR]", e)
        return None, None, "Error occurred during analysis."


# ── YIELD PREDICTION (UPDATED WITH MODEL) ──
def yield_prediction(crop, rainfall, pesticide, temperature):
    try:
        result = predict_yield(crop, rainfall, pesticide, temperature)
        return f"🌾 Predicted Yield: {result} tons/hectare"
    except Exception as e:
        print("[ERROR - YIELD]", e)
        return "Error in yield prediction"


# ── GRADIO UI ──
with gr.Blocks(title="Plant AI Assistant") as demo:

    gr.Markdown("# 🌿 Plant AI Assistant")

    with gr.Tabs():

        # ── TAB 1: LEAF ANALYSIS ──
        with gr.Tab("Leaf Analysis"):

            gr.Markdown(
                "Upload a leaf image. The system detects **biotic disease**, "
                "**abiotic stress**, or **healthy leaf** automatically."
            )

            image_input = gr.Image(type="numpy", label="Upload Leaf Image")
            run_button = gr.Button("🔍 Analyze", variant="primary")

            with gr.Row():
                original_image = gr.Image(label="Original Image")
                heatmap_output = gr.Image(label="GradCAM Heatmap")

            result_box = gr.Textbox(label="Analysis Results", lines=8)

            run_button.click(
                analyze_leaf,
                inputs=image_input,
                outputs=[original_image, heatmap_output, result_box],
            )

        # ── TAB 2: YIELD PREDICTION ──
        with gr.Tab("Yield Prediction"):

            crop = gr.Dropdown(
                ["Wheat", "Rice", "Maize", "Sugarcane"],
                label="Select Crop"
            )

            rainfall = gr.Slider(0, 3000, label="Rainfall (mm)")
            pesticide = gr.Slider(0, 500, label="Pesticide Usage")
            temperature = gr.Slider(0, 50, label="Temperature (°C)")

            yield_button = gr.Button("Predict Yield")
            yield_output = gr.Textbox(label="Yield Prediction")

            yield_button.click(
                yield_prediction,
                inputs=[crop, rainfall, pesticide, temperature],
                outputs=yield_output,
            )


demo.launch(share=True)