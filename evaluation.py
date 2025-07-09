import os
import pandas as pd
import re
import json
from glob import glob

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

def process_results(results_dir='results/qwen2_5_refo'):
    # Get all json files in the directory
    json_files = glob(os.path.join(results_dir, 'sample_*.json'))
    
    # Initialize lists to store data
    data = []

    evaluation_error = 0
    init_answer_error = 0
    # Process each json file

    error_idx_list = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
    
            ground_truth = result['ground_truth']
            final_answer = result['result']['final_answer']
            init_answer = result['result']['init_answer']
            extracted_option = extract_option_letter(final_answer)
            extracted_option_init = extract_option_letter(init_answer)
            extracted_option_eval = extract_option_letter(result['result']['evaluation'])

            if final_answer is not None: # 完整使用refo才需要分析
                if extracted_option_init == ground_truth and extracted_option != ground_truth: # 改错
                    # print(json_file, extracted_option_init, extracted_option, ground_truth)
                    evaluation_error += 1
                if extracted_option == ground_truth and extracted_option_init != ground_truth: # 改对
                    # print(json_file, extracted_option_init, extracted_option, ground_truth)
                    init_answer_error += 1

                if extracted_option_init != ground_truth and extracted_option != ground_truth: # 回答错误
                    # 提取回答错误样本的id
                    match = re.search(r"sample_(\d+)\.json", json_file)
                    if match:
                        id = match.group(1)
                        error_idx_list.append(int(id))

                    # if result['question_category'] in ["Landmark Position", "Association Reasoning"]:
                    #     print(json_file, extracted_option_init, extracted_option, ground_truth)
            # Create a row of data
            row = {
                'video_id': result['video_id'],
                'question': result['question'],
                'question_category': result['question_category'],
                'ground_truth': ground_truth,
                'model_answer': final_answer if final_answer else init_answer,
                'extracted_option': extracted_option_eval if final_answer else extracted_option_init,
            }
            data.append(row)
    
    print(error_idx_list)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate accuracy
    df['is_correct'] = df['extracted_option'] == df['ground_truth']
    
    # Initialize a list to store accuracy data for each question category
    accuracy_data = []
    
    # Group the DataFrame by the 'question_category' column
    for category, group in df.groupby('question_category'):
        correct_count = group['is_correct'].sum()
        accuracy = correct_count / len(group)
        accuracy_data.append({
            'Category': category,
            'Accuracy': accuracy,
            'Num': len(group)
        })
    
    # Calculate the overall accuracy across all categories
    total_correct = df['is_correct'].sum()
    total_accuracy = total_correct / len(df)
    accuracy_data.append({
        'Category': 'Total',
        'Accuracy': total_accuracy,
        'Num': len(df)
    })
    
    # Convert the accuracy data list into a DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Save results
    output_dir = results_dir + '/metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    # Save accuracy results to Excel file
    with pd.ExcelWriter(os.path.join(output_dir, 'accuracy_results.xlsx')) as writer:
        accuracy_df.to_excel(writer, index=False, sheet_name='Accuracy Results')
    
    print(f"改错数: {evaluation_error}")
    print(f"改对数: {init_answer_error}")
    print(f"Processed {len(json_files)} samples")
    print(f"Overall accuracy: {total_accuracy:.2%}")
    print("\nAccuracy by Category:")
    for _, row in accuracy_df.iterrows():
        if row['Category'] != 'Total':
            print(f"{row['Category']}: {row['Accuracy']:.2%} ({row['Num']} samples)")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    # results_dir = 'results/qwen2_5_only_reasoner'
    results_dir = 'results/qwen2_5_refo'
    process_results(results_dir) 