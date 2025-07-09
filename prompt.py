REASONER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions:
The template for the answer is: 
   Option: []; Reason: []
   where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option.
{question}

"""

# 改用gemini， 也要求以固定格式输出答案，如果两者答案一致跳过后续步骤
EVALUATOR_PROMPT = """ 
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. 
Here's a question about this video: {question}
Question Category and Intent: {question_category}

Please evaluate the following answer by: 
1. Independently analyze the question and provide your own answer
- Answer localization: Based on the question, locate the video segments that may contain the answer
- Analyze video segments: Carefully analyze the content in the video segments, reason step by step, retrieve relevant visual evidence, and provide your answer

2. As an evaluator, assess the correctness of the following answer based on your perspective:
- Answer to evaluate: {answer}
- If there are disagreements between this answer and your perspective, you need to point out the errors and provide reasons
- Be smart, logical, and critical in your evaluation.

Your revised answer must follow this exact format:
Location: []; Option: []; Reason: []
where:
- Location: 
  * Locate the video clips that may contain the answer to the question.
  * Must follow this format: Location: [(start_time, end_time), ...] e.g. [(0, 3), (32, 36)] means the answer is located between 0~3s and 32~36s of the video
- Option: Your own answer to the question, don't be influenced by the provided answer. Choose only one option from 'A' to 'E'
- Reason: 
  * Explain your choice based on the visual evidence in the video segments you located
  * If you believe the provided answer contains errors, please point them out and explain why
"""

FEEDBACKER_PROMPT = """
You are a viewpoint recorder who needs to document key disagreements and areas requiring further verification between reasoner's and critic's answer.

Question: {question}

Answer from Reasoner: {answer}

Answer and comments from Critic: {evaluation}

Based on the above answers, provide a structured analysis including:

1. Key Disagreements:
   - List specific points where the reasoner's answer and critic's answer differ
   - For each disagreement, clearly state both viewpoints
   - Highlight which aspects are supported by visual evidence and which are not

2. Required Verifications:
   - Identify specific visual evidence that needs to be re-examined
   - List additional frames or perspectives that would help resolve disagreements
   - Specify what kind of visual information would be needed to verify each point

3. Resolution Path:
   - Suggest specific steps to resolve each disagreement
   - Prioritize which verifications are most critical
   - Outline what additional evidence would be most helpful

Please ensure:
- Keep neutral and objective, only point out disagreements, do not give any subjective opinions or preferences
- Be specific about what visual evidence is needed
- Structure the analysis clearly and logically
- Do not give any subjective opinions or preferences
"""


OPTIMIZER_PROMPT = """
This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. 

Here's a question about this video: {question}
Question Category and Intent: {question_category}

This is your answer about this question: {answer}

However, the Evaluator has some disagreements with your perspective and has identified areas that need further verification:
{feedback}

Based on the above analysis, please:
1. Carefully review each identified disagreement
2. Re-examine the video frames, paying special attention to the specific visual evidence mentioned in the analysis
3. Consider the suggested verification steps and additional evidence needed
4. Provide a revised answer that addresses the identified issues
5. Note that the evaluator may make mistakes, so you need to carefully consider the feedback and independently think about the answer

Your revised answer must follow this exact format:
Option: []; Reason: []
where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. 
Reason explains why you choose this option.
""" 