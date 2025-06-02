from openai import OpenAI
from modelscope.msdatasets import MsDataset
import json, threading, time, os

API_KEY = 'YOUR_API_KEY'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'deepseek-r1'
THREAD = 12
SAMPLES = 1000
SYSTEM_PROMPT = '''
# 角色
你是一位小学教师，会耐心回答学生提出的数学问题

# 注意事项
- 回答要简洁明了，避免使用复杂的术语
- 如果问题太难，尝试简化问题，用简单的方式解释，容易让学生理解
- 如果问题不明确，可以适当引导学生澄清问题

# 风格
- 语气友好，鼓励学生提问
- 循序渐进，逐步引导学生理解问题
- 用生活中的例子来解释复杂的概念
- 经常确认学生是否理解你的回答；必要时通过反复解答确保学生理解
- 偶尔反问或提出有趣的问题，激发学生的思考

# 提问
{question}
'''


class R1Generator:
    def __init__(self, threads, dataset, samples):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.idx = 0
        self.threads = threads
        self.dataset = dataset
        self.samples = samples
        self.lock = threading.Lock()

    def generate(self, question):
        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", 
                 "content": SYSTEM_PROMPT.format(question=question)},
            ],
        #     temperature=0.0,
        #     max_tokens=1024,
        #     top_p=1,
        #     frequency_penalty=0,
        #    presence_penalty=0
        )
        return completion.choices[0].message.reasoning_content,\
                completion.choices[0].message.content
    
    def begin(self):
        self.idx = 0
        self.progress = 0
        self.results = [None] * self.samples
        self.threads_list = []
        for i in range(self.threads):
            t = threading.Thread(target=self.thread_main)
            t.start()
            self.threads_list.append(t)

    def join(self):
        while True:
            with self.lock:
                print(f'Progress: {self.progress}/{self.samples}', end='\r')
                if self.progress >= self.samples:
                    break
            time.sleep(1)
        for t in self.threads_list:
            t.join()
        return [res for res in self.results if res is not None]

    def thread_main(self):
        while True:
            with self.lock:
                if self.idx >= self.samples:
                    break
                curr_idx = self.idx
                self.idx += 1
            try:
                question = self.dataset[curr_idx]['question']
                reasoning, answer = self.generate(question)
                self.results[curr_idx] = {
                    'question': question,
                    'reasoning': reasoning,
                    'answer': answer
                }
            except Exception as e:
                pass
            with self.lock:
                self.progress += 1


if __name__ == '__main__':
    dataset = MsDataset.load('modelscope/gsm8k',
                             subset_name='main',
                             split='train',
                             trust_remote_code=True)
    r1 = R1Generator(threads=THREAD,
                     dataset=dataset,
                     samples=SAMPLES)
    r1.begin()
    results = r1.join()
    
    with open('distill_R1_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)