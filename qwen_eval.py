from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# load base model
model_name = 'Qwen/Qwen2.5-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Find latest checkpoint
checkpoints = os.listdir('./qwen_distill_output')
latest_checkpoint = sorted(filter(lambda x: x.startswith('checkpoint'), checkpoints), key=lambda x: int(x.split('-')[-1]))[-1]
lora_name = f'qwen_distill_output/{latest_checkpoint}'

SYSTEM_PROMPT = '''
# 任务
你现在是一位老师，回答学生提出的问题。

# 回答格式
<think>
逐步拆解、分析、反思难题，整理解答思路
</think>
以老师的身份向学生讲解问题
'''

def evaluate(model, query):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': '<think>'}  # Placeholder for model response
    ]
    text = tokenizer.apply_chat_template(messages, 
                                         tokenize=False,
                                         add_generation_prompt=False,
                                         continue_final_message=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4000,)
    completion_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return '<think>' + completion_text

query1 = 'strawberry里有几个r?'
query2 = '9.9和9.11哪个数更大?'

# Base model test
completion = evaluate(model, query1)
print(f"Base Model Completion:\n{completion}\n")
# LoRA model test
print(f'merge lora: {lora_name}')
model.load_adapter(lora_name)
completion = evaluate(model, query1)
print(f"LoRA Model Completion:\n{completion}\n")

