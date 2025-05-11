from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from dotenv import load_dotenv

import os

# Cargar variables del .env
load_dotenv()
print("DEBUG: HF_TOKEN =", os.getenv("HF_TOKEN"))
# Obtener token de variable de entorno
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("No se encontró la variable HF_TOKEN en el archivo .env")
else:
    login(token=HF_TOKEN)
# === CARGAR MODELO BASE DE GEMMA ===
model_id = "google/gemma-3-4b"
print(f"🔄 Descargando modelo {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Modelo {model_id} cargado correctamente en: {model.device}")