import json
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

SYSTEM_PROMPT = '''
# 任务
你现在是一位老师，回答学生提出的问题。

# 回答格式
<think>
逐步拆解、分析、反思难题，整理解答思路
</think>
以老师的身份向学生讲解问题
'''

def load_distill_dataset():
    dataset = {'messages': []}
    with open('distill_R1_results.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            sample = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': data['question']},
                {'role': 'assistant', 'content': f'<think>{data['reasoning']}</think>{data['answer']}'}
            ]
            dataset['messages'].append(sample)
    return Dataset.from_dict(dataset)

model_name = 'Qwen/Qwen2.5-3B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_distill_dataset()
sft_config = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    lr_scheduler_type='linear',
    warmup_ratio=.1,
    learning_rate=5e-6,
    max_seq_length=500,
    logging_steps=1,
    save_steps=.1,
    num_train_epochs=2,
    report_to='tensorboard',
    fp16=True,
    output_dir='./qwen_distill_output',
)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=['q_proj', 'k_proj', 'v_proj',
                    'o_proj', 'gate_proj', 'up_proj',
                    'down_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
)
trainer.train()