from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.pe1_base import PursuitEvadeGroup1Env
import os
import json
import time
import datetime
import csv
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import get_prompt


# === CONFIGURACIÓN ===
prompt_name = "p1:baselineCoT"  
system_prompt = get_prompt(prompt_name)
INTERVAL = 0.5  # segundos
MISSION_TIMEOUT = 240  # segundos

# === CARGA DEL MODELO GEMMA 3 ===
model_path = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)


class LLM1Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.env = None
        self.start_time = time.time()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        os.makedirs("logs", exist_ok=True)
        self.log_path = f"logs/llm1_run_{prompt_name}_{today}_{int(time.time())}.csv"
        self.log_file = open(self.log_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["t", "ft", "rt", "dt", "distance", "score"])

    def set_env(self, env):
        super().set_env(env)
        self.env = env

    def call_llm(self, observation):
        distance = observation["distance"]
        mission_time = observation["mission_time"]

        prompt = system_prompt + f"""

You are given the following observations:
- Pursuer position: {observation["pursuer_position"]}
- Pursuer velocity: {observation["pursuer_velocity"]}
- Evader position: {observation["evader_position"]}
- Evader velocity: {observation["evader_velocity"]}
- Vehicle mass: {observation["vehicle_mass"]} kg
- Remaining propellant (monopropellant): {observation["vehicle_propellant"]} kg
- Prograde vector: {observation["prograde"]}
- Mission time: {mission_time:.2f} seconds
- Current distance to evader: {distance:.2f} meters

Please plan and output your discrete thrust commands in JSON format (ft, rt, dt) and explain your reasoning.
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("\n--- LLM RESPONSE START ---")
        print(text)
        print("--- LLM RESPONSE END ---\n")

        match = re.search(r"\{[^}]*\"?ft\"?\s*:\s*-?\d\s*,[^}]*\"?rt\"?\s*:\s*-?\d\s*,[^}]*\"?dt\"?\s*:\s*-?\d\s*[^}]*\}", text)

        if match:
            return json.loads(match.group())
        else:
            print("No se encontró JSON válido. Usando acción nula.")
            return {"ft": 0, "rt": 0, "dt": 0}

    def distance(self, p1, p2):
        return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

    def get_action(self, observation):
        obs_dict = {
            "pursuer_position": observation[3:6],
            "pursuer_velocity": observation[6:9],
            "evader_position": observation[9:12],
            "evader_velocity": observation[12:15],
            "vehicle_mass": observation[1],
            "vehicle_propellant": observation[2],
            "prograde": [0, 0, 0], 
            "mission_time": observation[0],
        }

        distance = self.distance(obs_dict["pursuer_position"], obs_dict["evader_position"])
        obs_dict["distance"] = distance

        elapsed = time.time() - self.start_time
        if elapsed > MISSION_TIMEOUT:
            self.log_file.close()
            return {"burn_vec": [0, 0, 0, 0.1], "ref_frame": 0}

        action = self.call_llm(obs_dict)

        self.csv_writer.writerow([
            round(elapsed, 2),
            action["ft"], action["rt"], action["dt"],
            round(distance, 2),
            0
        ])
        self.log_file.flush()

        return {
            "burn_vec": [action["ft"], action["rt"], action["dt"], INTERVAL],
            "ref_frame": 0
        }

if __name__ == "__main__":
    agent = LLM1Agent()
    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=MISSION_TIMEOUT,
        debug=False
    )
    runner.run()
    print(f"Ejecución terminada. Log guardado en {agent.log_path}")
    
    obs_final = agent.env.get_observation()
    info = agent.env.get_info(obs_final, done=True)
    final_score = round(info.get("weighted_score", 0), 2)

    summary_dir = os.path.join("prompts_test", "prompt_metrics")
    os.makedirs(summary_dir, exist_ok=True)

    prompt_file_name = f"{prompt_name.replace(':', '_')}_scores.jsonl"
    prompt_summary_path = os.path.join(summary_dir, prompt_file_name)

    score_entry = {
        "prompt": prompt_name,
        "final_score": final_score,
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open(prompt_summary_path, "a") as f:
        f.write(json.dumps(score_entry) + "\n")

    print(f"Score final ({final_score}) guardado en {prompt_summary_path}")
