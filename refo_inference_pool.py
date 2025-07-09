
import json
import re
import os
import functools
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import EVALUATOR_PROMPT, FEEDBACKER_PROMPT, OPTIMIZER_PROMPT, REASONER_PROMPT
from torchvision import io
from multiprocessing import Pool

MAX_FRAMES = 32  # Maximum number of frames to read from the video
PARQUET_PATH = "data/UrbanVideo-Bench/MCQ.parquet"
OUTPUT_DIR = "results/qwen2_5_refo"

def get_total_frames(video_path: str) -> list:
    """Read video frames and convert to base64 using torchvision-like sampling."""
    video, _, _ = io.read_video(video_path, pts_unit="sec", output_format="TCHW")
    total_frames = video.size(0)
    return total_frames

def chat_with_model(model, processor, messages, local_rank=0):
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
    inputs = inputs.to(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

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

class Reasoner:
    def __init__(self, model, processor, local_rank=0):
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        
    def __call__(self, question: str, video_content) -> str:
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
        response = chat_with_model(self.model, self.processor, messages, self.local_rank)
        return response

class Evaluator:
    def __init__(self, model, processor, local_rank=0):
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        
    def __call__(self, question: str, answer: str, video_content, question_category: str) -> str:
        messages = [
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {
                            "type": "text",
                            "text": EVALUATOR_PROMPT.format(
                                question=question,
                                answer=answer,
                                question_category=question_category
                            )
                        },
                    ],
                }
            ]
        evaluation = chat_with_model(self.model, self.processor, messages, self.local_rank)
        return evaluation

class Feedbacker:
    def __init__(self, model, processor, local_rank=0):
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        
    def __call__(self, question: str, answer: str, evaluation: str, video_content) -> str:
        messages = [
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {
                            "type": "text",
                            "text": FEEDBACKER_PROMPT.format(
                                question=question,
                                answer=answer,
                                evaluation=evaluation
                            )
                        },
                    ],
                }
            ]
        feedback = chat_with_model(self.model, self.processor, messages, self.local_rank)
        return feedback

class Optimizer:
    def __init__(self, model, processor, local_rank=0):
        self.model = model
        self.processor = processor
        self.local_rank = local_rank
        
    def __call__(self, question: str, answer: str, feedback: str, video_content, question_category: str) -> str:
        messages = [
                {
                    "role": "user",
                    "content": [
                        video_content,
                        {
                            "type": "text",
                            "text": OPTIMIZER_PROMPT.format(
                                question=question,
                                answer=answer,
                                feedback=feedback,
                                question_category=question_category
                            )
                        },
                    ],
                }
            ]
        improved = chat_with_model(self.model, self.processor, messages, self.local_rank)

        return improved

class REFO:
    def __init__(self, local_rank=0):
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.local_rank = local_rank
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map={"": device}, cache_dir="./models",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        self.reasoner = Reasoner(self.model, self.processor, self.local_rank)
        self.evaluator = Evaluator(self.model, self.processor, self.local_rank)
        self.feedbacker = Feedbacker(self.model, self.processor, self.local_rank)
        self.optimizer = Optimizer(self.model, self.processor, self.local_rank)
        
    def process(self, question: str, video_path: str, question_category: str):
        total_frames = get_total_frames(video_path)
        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": 200704,
        }
        if total_frames >= MAX_FRAMES: # 限制视频帧数
            video_content["nframes"] = MAX_FRAMES

        print(f"local_rank:{self.local_rank}", flush=True)
        init_answer = self.reasoner(question, video_content)

        # print(f"Initial Answer: {init_answer[:100]}")
        init_option = extract_option_letter(init_answer)
        
        # Evaluate current answer
        evaluation = self.evaluator(question, init_answer, video_content, question_category)
        # print(f"Evaluation: {evaluation[:100]}")
        eval_option = extract_option_letter(evaluation)
        if init_option == eval_option: # 如果初始答案和评估答案一致，则直接返回
            return {
                "init_answer": init_answer,
                "evaluation": evaluation,
                "feedback": None,
                "final_answer": init_answer,
            }
        
        # Get feedback
        feedback = self.feedbacker(question, init_answer, evaluation, video_content)
        # print(f"Feedback: {feedback[:100]}...")
        
        # Optimize answer
        final_answer = self.optimizer(question, init_answer, feedback, video_content, question_category)
        # print(f"Final Answer: {final_answer[:100]}")
            
        return {
            "init_answer": init_answer,
            "evaluation": evaluation,
            "feedback": feedback,
            "final_answer": final_answer,
        } 
    
def process_chunk(chunk_args):
        chunk, local_rank = chunk_args
        refo = REFO(local_rank=local_rank)
        results = []
        print(f"[GPU {local_rank}] Start processing {len(chunk)} samples.", flush=True)
        for video_id, question, ground_truth, question_category, question_id in chunk[:2]:
            print(f"[GPU {local_rank}] Processing video_id: {video_id}, question_id: {question_id}", flush=True)
            video_path = f"data/UrbanVideo-Bench/videos/{video_id}"
            result = refo.process(question, video_path, question_category)
            output_path = os.path.join(OUTPUT_DIR, f"{question_id}.json")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "video_id": video_id,
                    "question": question,
                    "question_category": question_category,
                    "question_id": question_id,
                    'ground_truth': ground_truth,
                    'result': result
                }, f, indent=4)
            results.append(video_id)
        return results

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    n_gpus = torch.cuda.device_count()
    n_gpus = 2
    # 按GPU数分块
    data_chunks = [[] for _ in range(n_gpus)]
    for idx, row in enumerate(df.itertuples()):
        chunk_id = idx % n_gpus
        # 只保留基本类型，避免pickle问题
        data_chunks[chunk_id].append((
            row.video_id,
            row.question,
            row.answer,
            row.question_category,
            row.Question_id
        ))

    with Pool(n_gpus) as pool:
        results = pool.map(process_chunk, [(data_chunks[i], i) for i in range(n_gpus)])

    print("finished video_ids:\n", results)
