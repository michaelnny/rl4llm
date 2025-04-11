import sglang as sgl
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class DeterministicLogitProcessor(CustomLogitProcessor):
    """A dummy logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):

        print(logits.shape)

        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = 'token_id'

        # for i, param_dict in enumerate(custom_param_list):
        #     # Mask all other tokens
        #     logits[i, :] = -float("inf")
        #     # Assign highest probability to the specified token
        #     logits[i, param_dict[key]] = 0.0
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

    sampling_params = {'temperature': 0.8, 'top_p': 0.95}

    outputs = llm.generate(
        prompts,
        sampling_params,
        custom_logit_processor=DeterministicLogitProcessor().to_str(),
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
