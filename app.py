import os
# --- [수정 1] 사용할 물리적 GPU 지정 (2번, 3번) ---
# 이제 스크립트는 2번 GPU를 'cuda:0', 3번 GPU를 'cuda:1'로 인식합니다.
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

# --- [수정 2] 각 GPU에 대한 device 객체 생성 ---
# Vision 모델들은 cuda:0 (물리적 2번 GPU)에, Language 모델은 cuda:1 (물리적 3번 GPU)에 할당합니다.
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# --- GPU 사용 상태 체크 코드 ---
print("\n--- GPU Status Verification ---")
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    print("CUDA is not available or less than 2 GPUs are visible. Model parallelism is not possible.")
    # Fallback to single device if multi-GPU setup fails
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0
else:
    print(f"Script is configured to use physical GPUs #2 and #3.")
    print(f"PyTorch sees {torch.cuda.device_count()} visible GPU(s).")
    print(f"Device 0: {torch.cuda.get_device_name(0)} (Physical GPU #2)")
    print(f"Device 1: {torch.cuda.get_device_name(1)} (Physical GPU #3)")
print("-----------------------------\n")

from models.grit_model import DenseCaptioning
image_size = 384
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])


# --- [수정 3] 모델들을 각각 다른 GPU에 로드 ---
# Vision 모델들은 device0 (cuda:0)에 로드
print(f"Loading vision models on {device0}...")
model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
model.eval()
model = model.to(device0)
print("[INFO] initialize caption model success!")

intern_action = load_intern_action(device0)
trans_action = transform_action()
topil =  T.ToPILImage()
print("[INFO] initialize InternVideo model success!")

dense_caption_model = DenseCaptioning(device0)
dense_caption_model.initialize_model()
print("[INFO] initialize dense caption model success!")


# Language 모델(StableLM)은 device1 (cuda:1)에 로드
# 1단계에서 수정한 StableLMBot 클래스에 device를 전달합니다.
bot = StableLMBot()


# --- [수정 4] inference 함수 내에서 텐서를 올바른 GPU로 이동 ---
def inference(video_path, input_tag, progress=gr.Progress()):

    # Whisper
    progress(0.05, desc = "Transcribing Audio")
    transcript_text = transcribe_audio(video_path) # 추가
    
    data = loadvideo_decord_origin(video_path)
    progress(0.2, desc="Loading Videos")

    # Vision 모델용 텐서는 device0으로 보냅니다.
    action_index = np.linspace(0, len(data)-1, 8).astype(int)
    tmp,tmpa = [],[]
    for i,img in enumerate(data):
        tmp.append(transform(img).to(device0).unsqueeze(0)) # .to(device0)
        if i in action_index:
            tmpa.append(topil(img))
    action_tensor = trans_action(tmpa)
    TC, H, W = action_tensor.shape
    action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to(device0) # .to(device0)
    with torch.no_grad():
        prediction = intern_action(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]

    dense_caption = []
    dense_index = np.arange(0, len(data)-1, 5)
    original_images = data[dense_index,:,:,::-1]
    with torch.no_grad():
        for original_image in original_images:
            dense_caption.append(dense_caption_model.run_caption_tensor(original_image))
        dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])
    
    image = torch.cat(tmp).to(device0) # .to(device0)
    
    model.threshold = 0.68
    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        input_tag_list = None
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',',' | '))
    with torch.no_grad():
        caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
        progress(0.6, desc="Watching Videos")
        frame_caption = ' '.join([f"Second {i+1}:{j}.\n" for i,j in enumerate(caption)])
        if input_tag_list == None:
            tag_1 = set(tag_predict)
            tag_2 = ['none']
        else:
            _, tag_1 = model.generate(image,tag_input = None, max_length = 50, return_tag_predict = True)
            tag_2 = set(tag_predict)
        progress(0.8, desc="Understanding Videos")
        
    print(frame_caption, dense_caption)

    del data, action_tensor, original_image, image,tmp,tmpa
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return ' | '.join(tag_1),' | '.join(tag_2), frame_caption, dense_caption, gr.update(interactive = True), prediction, transcript_text

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

# ... 나머지 Gradio UI 코드는 동일 ...
# (이하 생략, 기존 코드와 동일)
with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
    gr.Markdown("<h1><center>Ask Anything with StableLM</center></h1>")
    gr.Markdown(
        """
        Ask-Anything is a multifunctional video question answering tool that combines the functions of Action Recognition, Visual Captioning and StableLM. Our solution generates dense, descriptive captions for any object and action in a video, offering a range of language styles to suit different user preferences. It supports users to have conversations in different lengths, emotions, authenticity of language.<br>   
        <p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_video_path = gr.Video(label="Input Video")
            input_tag = gr.Textbox(lines=1, label="User Prompt (Optional, Enter with commas)",visible=False)
          
            with gr.Row():
                with gr.Column(sclae=0.3, min_width=0):
                    caption = gr.Button("✍ Upload")
                    chat_video = gr.Button(" 🎥 Let's Chat! ", interactive=False)
                with gr.Column(scale=0.7, min_width=0):
                    loadinglabel = gr.Label(label="State")
        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbot", label="gpt")
            state = gr.State([])
            user_tag_output = gr.State("")
            image_caption_output = gr.State("")
            video_caption_output   = gr.State("")
            model_tag_output = gr.State("")
            dense_caption_output = gr.State("")
            with gr.Row(visible=False) as input_raws:
                with gr.Column(scale=0.8):
                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
                with gr.Column(scale=0.10, min_width=0):
                    run = gr.Button("🏃‍♂️Run")
                with gr.Column(scale=0.10, min_width=0):
                    clear = gr.Button("🔄Clear️")    
            with gr.Row():
                example_videos = gr.Dataset(components=[input_video_path], samples=[['images/playing_guitar.mp4'], ['images/yoga.mp4'], ['images/making_cake.mp4']])

    example_videos.click(fn=set_example_video, inputs=example_videos, outputs=[input_video_path])

    caption.click(lambda: gr.update(interactive = False), None, chat_video)
    caption.click(lambda: [], None, chatbot)
    caption.click(lambda: [], None, state)    
    caption.click(inference,[input_video_path,input_tag],[model_tag_output, user_tag_output, image_caption_output, dense_caption_output, chat_video, loadinglabel])

    chat_video.click(bot.init_agent, [image_caption_output, dense_caption_output, model_tag_output], [input_raws,chatbot])

    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)
    run.click(bot.run_text, [txt, state], [chatbot, state])
    run.click(lambda: "", None, txt)

    # clear.click(bot.memory.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)
    
demo.launch(server_name="0.0.0.0",enable_queue=True,)#share=True)
