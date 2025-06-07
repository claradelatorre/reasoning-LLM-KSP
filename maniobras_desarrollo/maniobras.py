from __future__ import annotations
from typing import List, Tuple
from kspdg.agent_api.base_agent import KSPDGBaseAgent
import math, json, os

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
DT_STEP = 1.0
DURATION = 240
RESULT_FILE = "maniobras_desarrollo/intercept_result.json"
LOG_FILE = "maniobras_desarrollo/intercept_log.json"

PLAN: List[Tuple[Tuple[float, float, float], float]] = [
    
   ((1, -0.015, 0.05), 30.0),
   ((0.00,  0.00,  0.00), 210.0)   
     # Final Burn - Align with Evader
   
]

class InterceptAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.idx = 0
        self.plan_start_time = None  # Tiempo de entrada en el paso actual
        self.log: List[dict] = []

    def get_action(self, observation):  # type: ignore[override]
        t_sim = observation[0]  # Tiempo simulado real

        # Extraer posiciones y velocidades
        px, py, pz = observation[3], observation[4], observation[5]
        ex, ey, ez = observation[9], observation[10], observation[11]
        pvx, pvy, pvz = observation[6], observation[7], observation[8]
        evx, evy, evz = observation[12], observation[13], observation[14]
        mono = observation[2]

        # Calcular distancia
        dx, dy, dz = ex - px, ey - py, ez - pz
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # Guardar log completo
        self.log.append({
            "t": t_sim,
            "distance": distance,
            "dx": dx,               
            "dy": dy,                
            "dz": dz,
            "mono": mono,
            "pursuer_velocity": {"x": pvx, "y": pvy, "z": pvz},
            "evader_velocity": {"x": evx, "y": evy, "z": evz}
        })

        # Si se ha acabado el plan
        if self.idx >= len(PLAN):
            return {"burn_vec": [0.0, 0.0, 0.0, DT_STEP], "ref_frame": 0}

        # Iniciar el tiempo del paso si aún no está
        if self.plan_start_time is None:
            self.plan_start_time = t_sim

        burn_vec, dur = PLAN[self.idx]
        elapsed = t_sim - self.plan_start_time

        # Avanzar al siguiente paso si se ha cumplido duración
        if elapsed >= dur:
            self.idx += 1
            self.plan_start_time = t_sim if self.idx < len(PLAN) else None
            return {"burn_vec": [0.0, 0.0, 0.0, DT_STEP], "ref_frame": 0}

        # Ejecutar paso actual
        return {"burn_vec": [*burn_vec, DT_STEP], "ref_frame": 0}

    def save_result(self):
        os.makedirs("pruebas", exist_ok=True)
        if not self.log:
            print("⚠️ No hay log para procesar.")
            return

        # Buscar entrada con menor distancia
        min_entry = min(self.log, key=lambda x: x["distance"])

        result = {
            "distance": min_entry["distance"],
            "time": min_entry["t"],
            "dX":            min_entry["dx"],     
            "dY":            min_entry["dy"],      
            "dZ":            min_entry["dz"],
            "pursuer_velocity": min_entry["pursuer_velocity"],
            "evader_velocity": min_entry["evader_velocity"],
            "final_mono": min_entry["mono"]
        }

        with open(RESULT_FILE, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Resultado guardado en {RESULT_FILE}")

    def save_log(self):
        os.makedirs("pruebas", exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"\n📒 Log completo guardado en {LOG_FILE}")

# ------------------------------------------------------------
if __name__ == "__main__":
    from kspdg.pe1.e1_envs import PE1_E1_I3_Env
    from kspdg.agent_api.runner import AgentEnvRunner

    agent = InterceptAgent()

    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=DURATION + 10,
        debug=True,
    )

    try:
        runner.run()
    finally:
        agent.save_log()
        agent.save_result()
