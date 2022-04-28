import gradio as gr
import degirum as dg

zoo=dg.connect_model_zoo()
model=zoo.load_model('efficientnet_em_imagenet--240x240_quant_n2x_cpu_1')

def infer(image):
    results=model(image)
    return results

inputs = gr.inputs.Image(label="Original Image")
outputs = gr.outputs.Textbox(label="Top5 Labels")

gr.Interface(infer, inputs, outputs).launch()
