import json
import math

# Archivo de entrada
INPUT_FILE = "maniobras_desarrollo/intercept_result.json"
# Valor de monopropelente al inicio de la misión (ajústalo tú)
INITIAL_MONO = 1200.0

def compute_relative_speed(pursuer: dict, evader: dict) -> float:
    dx = evader["x"] - pursuer["x"]
    dy = evader["y"] - pursuer["y"]
    dz = evader["z"] - pursuer["z"]
    return math.sqrt(dx**2 + dy**2 + dz**2)

def compute_score(d: float, v: float, f: float, t: float) -> float:
    return (0.1 * d)**2.0 + (0.5 * v)**1.5 + (0.1 * f)**1.25 + (0.01 * t)**1.0

if __name__ == "__main__":
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    d = data["distance"]
    t = data["time"]
    v = compute_relative_speed(data["pursuer_velocity"], data["evader_velocity"])
    f = INITIAL_MONO - data["final_mono"]

    score = compute_score(d, v, f, t)

    print("📊 Resultados:")
    print(f"  Distancia mínima (d):       {d:.2f} m")
    print(f"  Velocidad relativa (v):     {v:.2f} m/s")
    print(f"  Monopropelente usado (f):   {f:.2f} kg")
    print(f"  Tiempo hasta mínimo (t):    {t:.2f} s")
    print(f"\n✅ SCORE KSPDG: {score:.2f}")
