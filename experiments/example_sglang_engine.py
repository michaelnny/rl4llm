import sglang as sgl
import torch
from PIL import Image
from sglang.srt.conversation import chat_templates


def main():
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    llm = sgl.Engine(model_path=model_name)
    prompts = [
        'Hello, my name is',
        'The president of the United States is',
        'The capital of France is',
        'The future of AI is',
    ]

    sampling_params = {'temperature': 0.8, 'top_p': 0.95}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print('===============================')
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    torch.cuda.empty_cache()

    llm.release_memory_occupation()

    torch.cuda.empty_cache()

    print(torch.cuda.memory_allocated())


if __name__ == '__main__':
    main()
