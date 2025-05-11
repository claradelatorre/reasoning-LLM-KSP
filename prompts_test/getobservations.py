import krpc
import json
import os

def get_observation():
    # Conexión con kRPC
    conn = krpc.connect(name='ObservationExtractor')
    vessel = conn.space_center.active_vessel
    body = vessel.orbit.body
    ref_frame = body.non_rotating_reference_frame

    # Posición y velocidad del pursuer
    pursuer_pos = vessel.position(ref_frame)
    pursuer_vel = vessel.velocity(ref_frame)

    # Buscar el evader
    evader = next((v for v in conn.space_center.vessels if v.name == "Evader"), None)
    if evader is None:
        raise Exception("No se encontró el evader")

    # Posición y velocidad del evader
    evader_pos = evader.position(ref_frame)
    evader_vel = evader.velocity(ref_frame)

    # Prograde
    prograde = vessel.flight(ref_frame).prograde

    # Masa y monopropelente
    vehicle_mass = vessel.mass
    monopropellant = vessel.resources.amount("MonoPropellant")

    # Observaciones formateadas
    observation = {
        "vehicle_mass": vehicle_mass,
        "vehicle_propellant": monopropellant,
        "pursuer_pos_x": pursuer_pos[0],
        "pursuer_pos_y": pursuer_pos[1],
        "pursuer_pos_z": pursuer_pos[2],
        "pursuer_vel_x": pursuer_vel[0],
        "pursuer_vel_y": pursuer_vel[1],
        "pursuer_vel_z": pursuer_vel[2],
        "evader_pos_x": evader_pos[0],
        "evader_pos_y": evader_pos[1],
        "evader_pos_z": evader_pos[2],
        "evader_vel_x": evader_vel[0],
        "evader_vel_y": evader_vel[1],
        "evader_vel_z": evader_vel[2],
        "prograde": list(prograde),
        "mission_time": conn.space_center.ut
    }

    # Guardar en archivo JSON
    os.makedirs("observations", exist_ok=True)
    with open("observations/observation.json", "w") as f:
        json.dump(observation, f, indent=4)

    print("Observación guardada en observations/observation.json")
    return observation  # <- Añadido para reutilización

if __name__ == "__main__":
    get_observation()
