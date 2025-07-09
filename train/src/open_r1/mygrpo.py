# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from train.src.open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from math import isnan
import torch

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

# 无self-certainty的 暂时注释掉
def accuracy_reward(completions, answer, question, **kwargs):
    completion_content = [completion[0]["content"] for completion in completions] # 模型回复

    # 动态匹配任意数量的选项（A-Z）
    option_pattern = r"([A-Z])\. (.+)"
    choices = [dict(re.findall(option_pattern, ques)) for ques in question]

    rewards = []
    for content, correct_option, choice in zip(completion_content, answer, choices):
        content_match = re.search(r'<answer>(.*?)</answer>', content)

        # 从模型的回复中提取模型的答案
        model_answer = content_match.group(1).strip() if content_match else content.strip() # 提取模型回复中的答案
        model_option = model_answer.upper()

        # 确保模型答案完全匹配某个选项字母 否则为None
        if model_option not in choice.keys():
            model_option = None

        # 如果模型选择了正确答案，reward = 1.0，否则 reward = 0.0
        if model_option == correct_option:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"completion: {content}\n")
                if model_option is None:
                    f.write(f"model_answer: {model_answer}\n")
                else:
                    f.write(f"model_answer: {model_option}: {choice.get(model_option, 'Unknown option')}\n")
                f.write(f"ground_truth: {correct_option}: {choice.get(correct_option, 'Unknown option')}\n")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry two parts
reward_funcs_registry = {
    "format": format_reward,
    "accuracy": accuracy_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_urbanvideo_user_prompt(question, question_category):
    user_prompt = (
        "This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions:"
        f"{question}",
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer in form of the single option letter between the <answer> </answer> tags."
    )
    return "\n".join(user_prompt)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['format', 'accuracy']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # 数据读取
    if script_args.dataset_name == "data/UrbanVideo-Bench":
        print("data/UrbanVideo-Bench")
        dataset = load_dataset("parquet", data_files="data/UrbanVideo-Bench/MCQ.parquet")

    # 数据处理
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": SYSTEM_PROMPT + make_urbanvideo_user_prompt(example["question"], example["question_category"])},
                    ],
                },
            ],
        }
    dataset = dataset.map(make_conversation_video)  # Utilize multiprocessing for faster mapping

    # 数据划分
    # 划分为训练集和测试集（如 8:2） 数据的划分似乎不该在这里做？
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        cache_dir="./models/qwen2_5_vl",  # Specify cache directory for model loading
    )

    # Train and push the model to the Hub
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    script_args.dataset_train_split = "train"
    # print("training_args\n", training_args)
    # print("script_args\n", script_args)
    main(script_args, training_args, model_args)
