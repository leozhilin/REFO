import json
import re
import torch
import os
import cv2
import math
import base64
import pandas as pd
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import REASONER_PROMPT
from torchvision import io

MAX_FRAMES = 32  # Maximum number of frames to read from the video
PARQUET_PATH = "data/UrbanVideo-Bench/MCQ.parquet"
OUTPUT_DIR = "results/qwen2_5_only_reasoner"

def load_model():
    """Load the model and processor."""
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="./models",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor

def get_total_frames(video_path: str) -> list:
    """Read video frames and convert to base64 using torchvision-like sampling."""
    video, _, _ = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    total_frames = video.size(0)
    return total_frames

def chat_with_model(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def extract_option_letter(text):
    """Extract option letter from text."""
    if not isinstance(text, str):
        return None
    
    # Try to find option in the format "Option: X"
    match = re.search(r'Option:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1)
    
    # Try to find option in the format "Final Answer: X"
    match = re.search(r'Final Answer:\s*[\[\s]*(\w)', text)
    if match:
        return match.group(1)
    
    # If no match found, try to get the first letter
    return text[0].upper() if text else None

def save_result(output_path, video_id, question, question_category, answer, init_answer=None, evaluation=None, feedback=None, final_answer=None):
    """Save the result to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump({
            'video_id': video_id,
            'question': question,
            'question_category': question_category,
            'ground_truth': answer,
            'result': {
                "init_answer": init_answer,
                "evaluation": None,
                "feedback": None,
                "final_answer": None,
            }
        }, f, indent=2)
    print(f"Saved result to {output_path}")
if __name__ == "__main__":
    model, processor = load_model()
    
    df = pd.read_parquet(PARQUET_PATH)
    for idx, row in df.iterrows():
        if idx >= 314:
            video_id = row['video_id']
            question = row['question']
            answer = row['answer']
            question_category = row['question_category']
            Question_id = row['Question_id']
            video_path = f"data/UrbanVideo-Bench/videos/{video_id}"

            total_frames = get_total_frames(video_path)

            video_content = {
                "type": "video",
                "video": video_path,
                "max_pixels": 200704,
            }
            if total_frames >= MAX_FRAMES: # 限制视频帧数
                video_content["nframes"] = MAX_FRAMES
            print("Processing video:", video_id, "with frames:", min(total_frames, MAX_FRAMES))
            messages = [
                    {
                        "role": "user",
                        "content": [
                            video_content,
                            {
                                "type": "text",
                                "text": REASONER_PROMPT.format(question=question)
                            },
                        ],
                    }
                ]

            output_text = chat_with_model(model, processor, messages)
            print(extract_option_letter(output_text), answer)

            output_path = os.path.join(OUTPUT_DIR, f'sample_{idx}.json')
            save_result(output_path, video_id, question, question_category, answer, init_answer=output_text)