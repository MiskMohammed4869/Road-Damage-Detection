import os
import time
import cv2
import pandas as pd
import gradio as gr
from ultralytics import YOLO

# ==========================
# Load YOLO11 Model
# ==========================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = YOLO(MODEL_PATH)
class_names = model.names


# ==========================
# Image Detection Function
# ==========================
def detect_image(image, conf):

    start = time.time()

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False
    )

    result = results[0]

    annotated = result.plot()

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            score = float(box.conf[0])

            detections.append({
                "Damage Type": class_names[cls],
                "Confidence": f"{score:.2f}"
            })

    inference_time = (time.time() - start) * 1000

    df = pd.DataFrame(detections)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return (
        annotated,
        df,
        f"{inference_time:.2f} ms",
        len(detections)
    )


# ==========================
# Gradio Interface
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🚧 Road Damage Detection using YOLO11

Upload a road image to detect different types of road damage.

### Supported Damage Types
- Alligator Crack
- Longitudinal Crack
- Transverse Crack
- Pothole
- Block Crack
- Edge Crack
""")

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="numpy",
                label="Upload Road Image"
            )

            conf = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            detect_image_btn = gr.Button(
                "Detect Road Damage in Image",
                variant="primary"
            )

        with gr.Column():

            image_output = gr.Image(
                label="Detection Result"
            )

    gr.Markdown("## Detection Results")

    table = gr.Dataframe(
        headers=["Damage Type", "Confidence"],
        label="Detected Objects"
    )

    with gr.Row():

        inference = gr.Textbox(
            label="Inference Time"
        )

        total = gr.Number(
            label="Number of Detected Objects"
        )

    detect_image_btn.click(
        fn=detect_image,
        inputs=[image_input, conf],
        outputs=[
            image_output,
            table,
            inference,
            total
        ]
    )

demo.launch()