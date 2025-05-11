import os
import json
import time
import sys
import krpc
import csv
import datetime

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env

APPROACH_SPEED = 40
VESSEL_ACCELERATION = 0.1
EVASION_DISTANCE = 0
ROTATION_THRESHOLD = 0.03

class LLM2Agent(KSPDGBaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

        try:
            self.conn = krpc.connect()
            self.vessel = self.conn.space_center.active_vessel
            self.body = self.vessel.orbit.body
        except Exception as e:
            print("Exception while connecting to kRPC:", e)
            self.conn = None
            self.vessel = None
            self.body = None

        self.duration = 0.5
        self.scenario = os.environ.get("SCENARIO", "pe1_e1_i3").lower()
        self.model = os.environ.get("MODEL", "gpt-4.1-mini")

       

        with open("plan_p1_llm1.json", "r") as f:
            self.plan = json.load(f)

        self.log = self._setup_logger()

    def _setup_logger(self):
        today_date = datetime.datetime.now().strftime("%m-%d-%Y")
        logs_folder = os.path.join("training_data", f"logs_{today_date}")
        os.makedirs(logs_folder, exist_ok=True)
        log_path = os.path.join(logs_folder, f"llm2_log_{self.scenario}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
        log_file = open(log_path, mode='w', newline='')
        csv.writer(log_file).writerow(["ft", "rt", "dt", "duration"])
        return log_file

    def get_action(self, observation):
        ft = self.plan.get("ft", 0)
        rt = self.plan.get("rt", 0)
        dt = self.plan.get("dt", 0)

        burn_vec = [ft, rt, dt, self.duration]
        csv.writer(self.log).writerow([ft, rt, dt, self.duration])
        self.log.flush()

        time.sleep(1)
        return {
            "burn_vec": burn_vec,
            "ref_frame": 0
        }

def execute_agent():
    agent = LLM2Agent()
    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PE1_E1_I3_Env,
        env_kwargs={"loadfile": "persistent"},
        runner_timeout=240,
        debug=False
    )
    runner.run()

if __name__ == '__main__':
    execute_agent()
