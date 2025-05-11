# llm1_prueba.py
import openai
import os
import json
import re
from dotenv import load_dotenv
from prompts import get_prompt # Importa tu system prompt de prompts.py

prompt_name = "p1:baselineCoT"
system_prompt = get_prompt(prompt_name)# Cambia esto según el prompt que quieras usar
# Cargar clave de OpenAI desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cargar observaciones reales desde el archivo JSON
with open("observations/observation.json", "r") as f:
    observacion = json.load(f)

# Construcción del prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"""
You are given the following observations:
- Pursuer position: [{observacion["pursuer_pos_x"]}, {observacion["pursuer_pos_y"]}, {observacion["pursuer_pos_z"]}] m
- Pursuer velocity: [{observacion["pursuer_vel_x"]}, {observacion["pursuer_vel_y"]}, {observacion["pursuer_vel_z"]}] m/s
- Evader position: [{observacion["evader_pos_x"]}, {observacion["evader_pos_y"]}, {observacion["evader_pos_z"]}] m
- Evader velocity: [{observacion["evader_vel_x"]}, {observacion["evader_vel_y"]}, {observacion["evader_vel_z"]}] m/s
- Vehicle mass: {observacion["vehicle_mass"]} kg
- Remaining propellant (monopropellant): {observacion["vehicle_propellant"]} kg
- Prograde vector: {observacion["prograde"]}
- Mission time: {observacion.get("mission_time", 0)} seconds

Please plan and output your discrete thrust commands in JSON format (ft, rt, dt) and explain your reasoning.
"""}
]


# Llamada al modelo GPT-4 Turbo
response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=messages,
    temperature=0.3
)

# Mostrar respuesta
respuesta = response['choices'][0]['message']['content']
print("=== PLAN DE MANIOBRA ===")
print(respuesta)

# Extraer el bloque JSON con los comandos
match = re.search(r"\{[^}]*\"?ft\"?\s*:\s*-?1\s*,[^}]*\"?rt\"?\s*:\s*-?1\s*,[^}]*\"?dt\"?\s*:\s*-?1\s*[^}]*\}", respuesta) or \
        re.search(r"\{[^}]*\"?ft\"?\s*:\s*-?\d\s*,[^}]*\"?rt\"?\s*:\s*-?\d\s*,[^}]*\"?dt\"?\s*:\s*-?\d\s*[^}]*\}", respuesta)

if match:
    plan_dict = json.loads(match.group())
    with open("plan_p1_llm1.json", "w") as f:
        json.dump(plan_dict, f, indent=2)
    print("Plan guardado en 'plan_p1_llm1.json'")
else:
    print("No se pudo extraer un bloque JSON válido del plan generado.")
