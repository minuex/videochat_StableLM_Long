import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
import time
import numpy as np
from torch.nn import functional as F
import os
from scipy.spatial.distance import cdist
from util import load_video_segment 
from sentence_transformers import SentenceTransformer

start_message = """<|SYSTEM|># StableAssistant
- StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI.
- StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes.
- StableAssistant will refuse to participate in anything that could harm a human."""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StableLMBot:
    def __init__(self, device='cuda:0'):
        print(f"Initializing StableLM on device: {device}")
        self.device = device
        
        self.m = None
        self.tok = None
        self.generator = None
        self.messages = start_message
        self.video_index = []
        self.video_path = None
        
        self.tag2text_model = None
        self.vis_processor = None
        self.embedding_model = None
        self.dense_caption_model = None


    def load_model(self):
        if self.m is not None:
            return
            
        print(f"Starting to load the model to memory on {self.device}")
        model_name = "stabilityai/stablelm-tuned-alpha-3b"
        
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.generator = pipeline('text-generation', model=self.m, tokenizer=self.tok, device=self.m.device)
        print(f"Sucessfully loaded the model to the memory")


    def _find_best_scene(self, query_text):
        if not self.video_index:
            return None, "Video is not indexed yet."
        if self.embedding_model is None:
            return None, "Error: Embedding model is not initialized."


        print(f"\n--- Searching for query: '{query_text}' ---")
        query_vector = self.embedding_model.encode(query_text)
        
        index_vectors = np.array([item['vector'] for item in self.video_index])
        distances = cdist([query_vector], index_vectors, 'cosine')[0]
        
        best_scene_index = np.argmin(distances)
        best_scene = self.video_index[best_scene_index]
        
        print(f"Best match found: Time {best_scene['time']}s, Caption: '{best_scene['caption']}'")
        return best_scene, None

    def _get_detailed_description_for_scene(self, scene):
        print(f"--- Performing detailed analysis for scene at {scene['time']}s ---")
        segment_frames = load_video_segment(self.video_path, start_sec=scene['time'], duration_sec=30, fps=1)
        
        if len(segment_frames) == 0:
            return "Could not extract frames for the selected scene."

        captions = []
        with torch.no_grad():
            for frame_img in segment_frames:
                caption = self.dense_caption_model.run_caption_tensor(frame_img[:, :, ::-1])
                captions.append(caption)
        
        unique_captions = list(dict.fromkeys(captions))
        detailed_description = "In this scene, I can see: " + ", ".join(unique_captions)
        print(f"Detailed description: '{detailed_description}'")
        return detailed_description
        
    def init_agent(self, video_path, video_index, tag2text_model_list, vis_processor_list, embedding_model_list, dense_caption_model_list):
        self.load_model()
        
        self.video_path = video_path
        self.video_index = video_index
        
        self.tag2text_model = tag2text_model_list[0]
        self.vis_processor = vis_processor_list[0]
        self.embedding_model = embedding_model_list[0]
        self.dense_caption_model = dense_caption_model_list[0]
        
        self.messages = start_message
        
        initial_bot_message = "Video indexed. What would you like to know about it?"
        return gr.update(visible=True), [("Video uploaded and processed!", initial_bot_message)]

    def run_text(self, text, state):
        state.append([text, ""])

        best_scene, error = self._find_best_scene(text)
        if error:
            state[-1][-1] = error
            return state, state

        detailed_description = self._get_detailed_description_for_scene(best_scene)
        
        final_prompt = f"""<|SYSTEM|>{start_message}
<|USER|>I am asking a question about a video. I have found the most relevant scene for you.
The scene is about: "{best_scene['caption']}".
A more detailed observation of the scene reveals: "{detailed_description}".
Based on this information, please answer my question: {text}<|ASSISTANT|>"""
        
        output = self.generate(final_prompt)
        
        state[-1][-1] = output
        return state, state

    def generate(self, text):
        stop = StopOnTokens()
        result = self.generator(text, max_new_tokens=512, num_return_sequences=1, num_beams=1, do_sample=True,
                                 temperature=0.7, top_p=0.95, top_k=500, stopping_criteria=StoppingCriteriaList([stop]))
        generated_text = result[0]["generated_text"]
        answer = generated_text.split("<|ASSISTANT|>")[-1]
        return answer.strip()
