import gradio as gr
import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread

HISTORY_SIZE = 3
SYSTEM_PROMPT = '''
# 任务
你现在是一位老师，回答学生提出的问题。

# 回答格式
<think>
逐步拆解、分析、反思难题，整理解答思路
</think>
以老师的身份向学生讲解问题
'''

def chat_stream(model_selector, query, history):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
    ]

    for query_, answer_ in history:
        messages.append({'role': 'user', 'content': query_})
        messages.append({'role': 'assistant', 'content': answer_})

    messages.append({'role': 'user', 'content': query})
    messages.append({'role': 'assistant', 'content': '<think>'})  # Placeholder for model response
    text = tokenizer.apply_chat_template(messages, 
                                         tokenize=False,
                                         add_generation_prompt=False,
                                         continue_final_message=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)
    
    if model_selector == 'Qwen Base':
        model.disable_adapters()
    else:
        model.enable_adapters()

    streamer = TextIteratorStreamer(tokenizer,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=2000) 
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text
    thread.join()


with gr.Blocks(css='.qwen-logo img {height:200px; wdith:600px; margin:0 auto;}') as demo:
    with gr.Row():
        chatbot = gr.Chatbot(label='Deepseek R1 Distilled Qwen Chatbot', show_copy_button=True)
    with gr.Row():
        model_selector = gr.Dropdown(choices=['Qwen Base', 'Qwen Distill'],
                                     label='Select Model')
    with gr.Row():
        query_box = gr.Textbox(label='Enter your query here',
                               autofocus=True,
                               lines=2,)
    with gr.Row():
        clear_btn = gr.ClearButton([query_box, chatbot], value='Clear History')
        submit_btn = gr.Button(value='Submit')

    def chat(model_selector, query, history):
        full_response = '<think>'
        replace_response = ''
        for response in chat_stream(model_selector=model_selector,
                                    query=query,
                                    history=history):
            full_response += response
            replace_response = full_response.replace('<think>', '[开始思考]\n').replace('</think>', '\n[思考结束]\n')
            yield '', history + [(query, replace_response)]
        history.append((query, replace_response))
        while len(history) > HISTORY_SIZE:
            history.pop(0)
    
    submit_btn.click(
        chat,
        inputs=[model_selector, query_box, chatbot],
        outputs=[query_box, chatbot],
        show_progress=True,
    )

if __name__ == "__main__":
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Find latest checkpoint
    checkpoints = os.listdir('./qwen_distill_output')
    latest_checkpoint = sorted(filter(lambda x: x.startswith('checkpoint'), checkpoints), key=lambda x: int(x.split('-')[-1]))[-1]
    lora_name = f'qwen_distill_output/{latest_checkpoint}'

    model.load_adapter(lora_name)

    demo.queue(200)
    demo.launch(server_name='0.0.0.0', max_threads=500)