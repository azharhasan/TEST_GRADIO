import gradio as gr
from fastai.vision.all import *
from fastbook import *
import pathlib

plt = platform.system()
if plt != 'Windows': pathlib.WindowsPath = pathlib.PosixPath

ln = load_learner("model.pkl")
catagories = ("Dog", "Cat")

def classify_image(img):
    pred, idx, probs = ln.predict(img)
    return dict(zip(catagories, map(float, probs)))

intf = gr.Interface(fn=classify_image, inputs=gr.Image(), outputs=gr.Label())
intf.launch(inline=False)

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
