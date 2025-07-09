conda create -n refo python=3.11 -y
conda activate refo


pip install transformers trl lighteval -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install qwen-vl-utils[decord]==0.0.8 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install wandb -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple/



pip install --upgrade numpy scipy -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

## 研究计划
1. 基础配置
- 配置好基本环境并记录md（to do：暂时用的是Visual-RFT的环境，以上配置方法flash-attn会出问题，必须要记录到一个完整容易复现的环境记录）
- 准备好数据集（done）
- 下载好模型文件（done）
- 确保qwen2.5-vl-3b官方脚本可正常运行（done）
- 确保qwen2.5-vl-3b在自己的数据集上能正常运行（done）
    - 改成并行的，提高效率(to do 遇到了些bug，伪并行，回头有空再弄)
    - 考虑用vllm加速？

2. 实现基本REFO框架
- 编写好四套提示词（done）
- 按REFO框架搭建训练代码（done）
- 在UrbanVideo-Bench上跑测REFO，并收集实验结果（to do：在表格中系统统计记录数据）
- 提示词格式改成<></>再测一遍，确保不会影响效果

3. 实现RL or SFT的训练框架，能够在UrbanVideo-Bench上训练qwen2.5-vl-3b并能取得正向效果
- 基于视频的GRPO训练算法跑通（done）
- 基本规则奖励设计（to do：还差准确率奖励）
- 跑通vllm版本，充分利用8卡显存，确保起码支持32帧输入和4次重复采样（to do）
- 跑测训练完的reasoner的效果

4. 针对推理、评估、反馈、优化收集训练数据，制作四个训练数据集，训练四个能力侧重不同的LLM

5. 完整版REFO跑测，并和之前的结果及其他工作的结果进行对比实验

6. 尝试base model 和 adapter的拆分，减轻推理代价



暂时要这样才能正常运行，因为visual-rft是在原来的环境下运行的，会指向原来的open_r1目录
export PYTHONPATH=$(pwd):$PYTHON
