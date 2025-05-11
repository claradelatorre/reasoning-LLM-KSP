from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """<start_of_turn>user
What is a good place for travel in the US?<end_of_turn>
<start_of_turn>model
"""

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)
print("Modelo cargado:", model.config._name_or_path)

text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nRespuesta generada:\n")
print(text)
