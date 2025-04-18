import sglang as sgl
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class SglExploreLogitProcessor(CustomLogitProcessor):

    def __init__(self, explore_steps, skip_n, explore_top_k, decay=0.9):
        super().__init__()
        self.explore_steps, self.skip_n = explore_steps, skip_n
        self.explore_top_k, self.decay = explore_top_k, decay

    def __call__(self, logits, custom_param_list):
        import torch

        assert logits.shape[0] == len(custom_param_list)

        bsz, vocab = logits.shape

        temps = []
        for row in range(bsz):
            cfg = custom_param_list[row]  # ← one dict per live sequence
            step = cfg['step']

            # exploration …
            if self.skip_n <= step < self.skip_n + self.explore_steps:
                k = max(
                    2,
                    int(
                        self.explore_top_k * self.decay ** (step - self.skip_n)
                    ),
                )
                idx = logits[row].topk(min(k, vocab)).indices
                logits[row].fill_(-1e6)
                logits[row][idx] = 1000.0  # uniform mass
                print(f"exploring start {k}")

            cfg['step'] = step + 1  # persist
            temps.append(cfg['temperature'])

        logits.div_(torch.tensor(temps, device=logits.device).unsqueeze(1))
        return logits


def main():
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    llm = sgl.Engine(
        model_path=model_name,
        enable_custom_logit_processor=True,
        enable_memory_saver=True,
    )
    prompts = [
        'Hello, my name is',
        'The president of the United States is',
        'The capital of France is',
        'The future of AI is',
    ]

    sampling_params = []
    for t in [0, 0.3, 0.6, 0.9]:
        sp = {
            'temperature': 1.0,  # <‑ no double scaling
            'top_p': 0.95,
            'top_k': -1,  # keep your explore mask
            'custom_params': {'temperature': t, 'step': 0},
        }
        sampling_params.append(sp)

    logit_processor = SglExploreLogitProcessor(
        explore_steps=5,
        explore_top_k=100,
        skip_n=0,
        decay=0.8,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        custom_logit_processor=logit_processor.to_str(),
    )
    for prompt, output in zip(prompts, outputs):
        print('===============================')
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    torch.cuda.empty_cache()

    llm.release_memory_occupation()

    torch.cuda.empty_cache()

    print(torch.cuda.memory_allocated())


if __name__ == '__main__':
    main()
