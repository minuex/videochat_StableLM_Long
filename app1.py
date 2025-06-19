import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
print("Attempting to use physical GPU #2 and #3...")

import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
from stablelm import *
from load_internvideo import *
from sentence_transformers import SentenceTransformer

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device0)

print("\n--- GPU Status Verification ---")
if not torch.cuda.is_available():
    print("CUDA is not available. The program will run on CPU.")
else:
    print(f"PyTorch sees {torch.cuda.device_count()} visible GPU(s).")
    if torch.cuda.device_count() > 0: print(f"Device 0: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1: print(f"Device 1: {torch.cuda.get_device_name(1)}")
print("-----------------------------\n")

from models.grit_model import DenseCaptioning

image_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
vis_processor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])

print(f"Loading vision models on {device0}...")
tag2text_model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b')
tag2text_model.eval()
tag2text_model = tag2text_model.to(device0)
tag2text_model.device = device0
print("[INFO] initialize caption model success!")

dense_caption_model = DenseCaptioning(device0)
dense_caption_model.initialize_model()
print("[INFO] initialize dense caption model success!")

print(f"Loading Korean sentence embedding model on {device0}...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device0)
print("[INFO] initialize sentence embedding model success!")

# 챗봇 객체 생성
bot = StableLMBot(device=device1)


def inference_and_index(video_path, progress=gr.Progress()):
    if video_path is None:
        return None, None, gr.update(interactive=False), "Please upload a video first.", []

    progress(0.1, desc="Loading Video & Indexing...")
    data = loadvideo_decord_origin(video_path)
    
    video_index = []
    frame_interval = 30
    dense_index = np.arange(0, len(data) - 1, frame_interval)
    original_images = data[dense_index, :, :, ::-1]

    with torch.no_grad():
        for i, original_image in enumerate(original_images):
            timestamp_sec = dense_index[i]
            caption_text = dense_caption_model.run_caption_tensor(original_image)
            caption_vector = embedding_model.encode(caption_text, convert_to_tensor=True)
            video_index.append({
                'time': timestamp_sec,
                'caption': caption_text,
                'vector': caption_vector.cpu().numpy()
            })

    print("--- Indexing Finished ---")
    progress(0.8, desc="Indexing complete. Ready to chat.")

    del data, original_images
    torch.cuda.empty_cache()
    
    return video_path, video_index, gr.update(interactive=True), "Indexing complete. Click 'Let's Chat!'"

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
    gr.Markdown("<h1><center>Ask Anything with StableLM</center></h1>")
    
    gr.Markdown(
        """
        Ask-Anything is a multifunctional video question answering tool that combines the functions of Action Recognition, Visual Captioning and StableLM. Our solution generates dense, descriptive captions for any object and action in a video, offering a range of language styles to suit different user preferences. It supports users to have conversations in different lengths, emotions, authenticity of language.<br>   
        <p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """
    )
    
    state = gr.State([])
    video_path_state = gr.State()
    video_index_state = gr.State([])
    
    tag2text_model_state = gr.State(value=[tag2text_model])
    vis_processor_state = gr.State(value=[vis_processor])
    embedding_model_state = gr.State(value=[embedding_model])
    dense_caption_model_state = gr.State(value=[dense_caption_model])

    with gr.Row():
        with gr.Column():
            input_video_path = gr.Video(label="Input Video")
            with gr.Row():
                with gr.Column(scale=0.3, min_width=0):
                    upload_button = gr.Button("✍ Upload & Index Video")
                    chat_video_button = gr.Button("🎥 Let's Chat!", interactive=False)
                with gr.Column(scale=0.7, min_width=0):
                    status_label = gr.Label(label="State")
        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbot", label="StableLM Video Assistant")
            with gr.Row(visible=False) as input_raws:
                with gr.Column(scale=0.8):
                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                with gr.Column(scale=0.10, min_width=0):
                    run_button = gr.Button("🏃‍♂️Run")
                with gr.Column(scale=0.10, min_width=0):
                    clear_button = gr.Button("🔄Clear️")    
            with gr.Row():
                example_videos = gr.Dataset(components=[input_video_path], samples=[['images/playing_guitar.mp4'], ['images/yoga.mp4'], ['images/making_cake.mp4']])

    example_videos.click(fn=set_example_video, inputs=example_videos, outputs=[input_video_path])

    upload_button.click(
        fn=inference_and_index,
        inputs=[input_video_path],
        outputs=[video_path_state, video_index_state, chat_video_button, status_label]
    )

    chat_video_button.click(
        fn=bot.init_agent,
        inputs=[video_path_state, video_index_state, tag2text_model_state, vis_processor_state, embedding_model_state, dense_caption_model_state],
        outputs=[input_raws, chatbot]
    )

    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    run_button.click(bot.run_text, [txt, state], [chatbot, state])
    run_button.click(lambda: "", None, txt)
    
    def clear_all():
        return [], [], None, gr.update(visible=False), "Chat cleared."

    clear_button.click(clear_all, [], [chatbot, state, video_path_state, input_raws, status_label])

demo.launch(server_name="0.0.0.0", enable_queue=True)
