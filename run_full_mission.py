import os
import json
import time
import krpc
import shutil
import datetime
from mission_generator import extract_observation
import llm1_temporal
import llm2_act 
# Rutas
MISSION_FOLDER = r"C:\Users\Clara\Desktop\TFM\KSP_win64\saves\missions\pe1_i3"
RESET_FILE = os.path.join(MISSION_FOLDER, "reset.missionsfs")
PERSISTENT_FILE = os.path.join(MISSION_FOLDER, "persistent.sfs")

# 1. Reiniciar misión
def reset_mission():
    if os.path.exists(RESET_FILE):
        shutil.copyfile(RESET_FILE, PERSISTENT_FILE)
        print("Misión reiniciada correctamente.")
    else:
        raise FileNotFoundError(f"No se encontró el archivo: {RESET_FILE}")



def generate_observation():
    print("🔍 Extrayendo observaciones desde KSP...")
    extract_observation()
    print("✅ Observaciones guardadas en observation.json.")



# Ejecutar pipeline completo
if __name__ == "__main__":
   
    generate_observation()
    llm1_temporal
    llm1_temporal
    reset_mission()