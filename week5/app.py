#build gradio app
from pathlib import Path
import sys
import gradio as gr
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.predict import predict_gradio

image = gr.Image()
label = gr.Label(num_top_classes=5)

gradio = gr.Interface(fn=predict_gradio, inputs=image, outputs=label, title="Pokemon classifier")
gradio.launch(debug='True', share=True)