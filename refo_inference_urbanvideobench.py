
import json
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import EVALUATOR_PROMPT, FEEDBACKER_PROMPT, OPTIMIZER_PROMPT, REASONER_PROMPT
from torchvision import io
MAX_FRAMES = 32  # Maximum number of frames to read from the video
PARQUET_PATH = "data/UrbanVideo-Bench/MCQ.parquet"
OUTPUT_DIR = "results/qwen2_5_refo"

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

class Reasoner:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
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
        response = chat_with_model(self.model, self.processor, messages)
        return response

class Evaluator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
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
        evaluation = chat_with_model(self.model, self.processor, messages)
        return evaluation

class Feedbacker:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
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
        feedback = chat_with_model(self.model, self.processor, messages)
        return feedback

class Optimizer:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
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
        improved = chat_with_model(self.model, self.processor, messages)

        return improved

class REFO:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="./models",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        self.reasoner = Reasoner(self.model, self.processor)
        self.evaluator = Evaluator(self.model, self.processor)
        self.feedbacker = Feedbacker(self.model, self.processor)
        self.optimizer = Optimizer(self.model, self.processor)
        
    def process(self, question: str, video_path: str, question_category: str):
        total_frames = get_total_frames(video_path)
        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": 200704,
        }
        if total_frames >= MAX_FRAMES: # 限制视频帧数
            video_content["nframes"] = MAX_FRAMES


        init_answer = self.reasoner(question, video_content)
        print(f"Initial Answer: {init_answer[:100]}")
        init_option = extract_option_letter(init_answer)
        
        # Evaluate current answer
        evaluation = self.evaluator(question, init_answer, video_content, question_category)
        print(f"Evaluation: {evaluation[:100]}")
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
        print(f"Feedback: {feedback[:100]}...")
        
        # Optimize answer
        final_answer = self.optimizer(question, init_answer, feedback, video_content, question_category)
        print(f"Final Answer: {final_answer[:100]}")
            
        return {
            "init_answer": init_answer,
            "evaluation": evaluation,
            "feedback": feedback,
            "final_answer": final_answer,
        } 
    
if __name__ == "__main__":
    refo = REFO()
    
    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    
    for index, row in df.iterrows():
        video_id = row['video_id']
        question = row['question']
        ground_truth = row['answer']
        question_category = row['question_category']
        Question_id = row['Question_id']
        video_path = f"data/UrbanVideo-Bench/videos/{video_id}"

        
        result = refo.process(question, video_path, question_category)
        
        output_path = os.path.join(OUTPUT_DIR, f"sample_{index}.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "video_id": video_id,
                "question": question,
                "question_category": question_category,
                'ground_truth': ground_truth,
                'result': result
            }, f, indent=4)


            
