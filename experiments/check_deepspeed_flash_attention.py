import deepspeed
from transformers import AutoModelForCausalLM

model_name = 'Qwen/Qwen2.5-0.5B'

# Load the HF model
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
print('Before DeepSpeed:')
print(model.config)  # Look for attention-related settings
print(model.model.layers[0].self_attn)  # Inspect the attention layer

# Initialize DeepSpeed
ds_config = {'zero_optimization': {'stage': 1}}  # Your ZeRO-1/2 config
engine, _, _, _ = deepspeed.initialize(model=model, config_params=ds_config)

print('After DeepSpeed:')
print(model.config)  # Should be unchanged
print(model.model.layers[0].self_attn)  # Inspect the attention layer
