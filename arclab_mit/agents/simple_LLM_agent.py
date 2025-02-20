# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
This script is a "Hello World" for writing agents that can interact with
a KSPDG environment.

Instructions to Run:
- Start KSP game application.
- Select Start Game > Play Missions > Community Created > pe1_i3 > Continue
- In kRPC dialog box click Add server. Select Show advanced settings and select Auto-accept new clients. Then select Start Server
- In a terminal, run this script

"""
import os
import openai
import json
import time
import numpy as np
import sys
import traceback
import httpx
import re
import csv
import datetime
import argparse
from termcolor import colored

import random
import krpc
from openai import OpenAI

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env

from kspdg.lbg1.lg2_envs import LBG1_LG2_I2_Env

from os.path import join, dirname
from dotenv import load_dotenv

from kspdg.sb1.e1_envs import SB1_E1_I1_Env, SB1_E1_I2_Env, SB1_E1_I3_Env, SB1_E1_I4_Env, SB1_E1_I5_Env
from arclab_mit.agents.agent_common import set_env_paths, setup_scenarios, State, Action, orbital_from_vectors

# CONSTANTS
kerbin_mu = 3.5316000e12   # Gravitational parameter of Kerbin (m^3/s^2)
kerbin_radius = 600000     # Radius of Kerbin (m)
kerbin_mass = 5.2915158e22 # Mass of Kerbin (kg)

# Distance thresholds for guard evasive action
GUARD_DISTANCE_HIGH = 1200
GUARD_DISTANCE_LOW = 900

# Load configuration from .env
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Load prompts from alex_prompts.txt into environment variables
prompts_path = join(dirname(__file__), 'alex_prompts.txt')
load_dotenv(prompts_path)

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def load_prompts_from_file(file_name):
    """
    Reads prompts from a file and extracts the relevant PE problem prompts.
    Args:
        file_name: Name of the file containing prompts (e.g., 'alex_prompts.txt').
    Returns:
        A dictionary with the extracted prompts for the PE problem.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    prompts = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Flags to identify the relevant section
    pe_section = False
    for line in lines:
        line = line.strip()
        if "Prompts for PE problem" in line:
            pe_section = True
        elif "Prompts for SB problem" in line:  # Exit the PE section
            pe_section = False

        # Extract prompts
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            prompts[key] = value

    return prompts


def load_prompts():
    """
    Load prompts from the default location relative to this file.
    Returns:
        dict: The loaded prompts
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file_name = "alex_prompts.txt"
    return load_prompts_from_file(os.path.join(script_dir, prompt_file_name))


class LlamaAPIClient:
    def __init__(self, base_url: str, timeout=60):
        self.base_url = base_url
        self.timeout = timeout

    def get(self, endpoint: str, params: dict = None):
        response = httpx.get(self.base_url + endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()  # Raises an error for bad responses
        return response.json()

    def post(self, endpoint: str, data: dict = None, json: dict = None, files: dict = None):
        response = httpx.post(self.base_url + endpoint, data=data, json=json, timeout=self.timeout, files=files)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint: str, data: dict = None, json: dict = None):
        response = httpx.put(self.base_url + endpoint, data=data, json=json, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def delete(self, endpoint: str):
        response = httpx.delete(self.base_url + endpoint)
        response.raise_for_status()
        return response.json()

    def close(self):
        httpx.aclose()


class SimpleLLMAgent(KSPDGBaseAgent):
    def __init__(self, prompts=None, log_file_prefix="simple_LLM_agent_log", **kwargs):
        super().__init__()

        # Process kwargs
        log = kwargs.get("arg1", False)

        # If prompts not provided, load them from default location
        self.prompts = prompts if prompts is not None else load_prompts()
        self.is_claude = kwargs.get('is_claude', False)
        self.is_few_shot = kwargs.get('is_few_shot', False)
        
        # Configure functions/tools based on LLM type
        if self.is_claude:
            self.tools = [{
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ft": {
                            "type": "string",
                            "enum": ["forward", "backward", "none"],
                            "description": "The forward throttle direction."
                        },
                        "rt": {
                            "type": "string",
                            "enum": ["right", "left", "none"],
                            "description": "The right throttle direction."
                        },
                        "dt": {
                            "type": "string",
                            "enum": ["up", "down", "none"],
                            "description": "The down throttle direction."
                        }
                    },
                    "required": ["ft", "rt", "dt"]
                }
            }]
        else:
            self.functions = [{
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ft": {
                            "type": "string",
                            "enum": ["forward", "backward", "none"],
                            "description": "The forward throttle direction.",
                        },
                        "rt": {
                            "type": "string",
                            "enum": ["right", "left", "none"],
                            "description": "The right throttle direction.",
                        },
                        "dt": {
                            "type": "string",
                            "enum": ["up", "down", "none"],
                            "description": "The down throttle direction.",
                        },
                    },
                    "required": ["ft", "rt", "dt"],
                },
            }]
        self.first_completion = True
        self.closest_distance = sys.float_info.max
        self.use_relative_coordinates = (os.environ['USE_RELATIVE_COORDINATES'].lower() == "true")
        self.use_short_argument_names = (os.environ['USE_SHORT_ARGUMENT_NAMES'].lower() == "true")
        self.use_prograde = True  # Enable prograde calculations
        self.use_cot = True       # Enable Chain of Thought
        self.use_cot_speed_limit = False  # Disable speed limit considerations

        # Connect to the KRPC server
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.body = self.vessel.orbit.body

        # Load model and scenario configuration
        self.model = os.environ['BASE_MODEL']
        self.scenario = os.environ['SCENARIO']
        self.ignore_time = os.environ['IGNORE_TIME']

        # Select LLM client: gpt or llama
        self.use_llama = (os.environ['USE_LLAMA'].lower() == "true")
        if self.use_llama:
            # HACK TO BE REMOVED
            if self.scenario.startswith("LBG"):
                self.scenario = "PE"

        # Set prompts
        self.system_prompt = "You are an autonomous agent."
        self.user_prompt_template = "{obs} What is the best throttle to capture the evader?"
        self.cot_prompt = ""
        if self.scenario.startswith("PE"):
            self.system_prompt = os.environ["PE_SYSTEM_PROMPT"]
            self.user_prompt_template = self.prompts.get("PE_USER_PROMPT", "{obs} What is the best throttle to capture the evader?")
            self.cot_prompt = os.environ["PE_CHAIN_OF_THOUGHT"]
        elif self.scenario.startswith("SB"):
            self.system_prompt = os.environ["SB_SYSTEM_PROMPT"]
            self.user_prompt_template = self.prompts.get("SB_USER_PROMPT", "{obs} What is the best throttle to capture the evader?")
            self.cot_prompt = os.environ["SB_CHAIN_OF_THOUGHT"]
        elif self.scenario.startswith("LBG"):
            print("is claude: " + str(self.is_claude))
            if self.is_claude:
                self.system_prompt = os.environ["CLAUDE_VISION_SYSTEM_PROMPT"]
            elif self.is_few_shot:
                self.system_prompt = os.environ["LBG_SYSTEM_PROMPT_VISION_FEW_SHOT"]
            else:
                print("Entering LBG scenario")
                self.system_prompt = os.environ["LBG_SYSTEM_PROMPT_VISION"]
            self.user_prompt_template = self.prompts.get("LBG_USER_PROMPT", "{obs} What is the best throttle to capture the lady?")
            self.cot_prompt = os.environ["LBG_CHAIN_OF_THOUGHT"]

        # FastAPI client for LlaMa
        base_url = os.environ['LLAMA_URL']

        # Set client
        self.client = OpenAI()
        if self.use_llama:
            self.client = LlamaAPIClient(base_url)

        # Set default duration for actions
        self.duration = 0.5

        # Set use_enum
        self.use_enum = True

        # Create streams to log actions
        if not log:
            self.log_csv = None
            self.log_jsonl = None
        else:
            os.makedirs("logs", exist_ok=True)
            log_name = "./logs/" + log_file_prefix + "_" + self.scenario + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
            self.log_csv = open(log_name, mode='w', newline='')
            if self.scenario.lower().startswith("lbg"):
                head = ['time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                        'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                        'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                        'evader_vel_z', 'guard_pos_x', 'guard_pos_y', 'guard_pos_z', 'guard_vel_x', 'guard_vel_y',
                        'guard_vel_z', "pursuer_orbital", "evader_orbital", "guard_orbital"]
            else:
                head = ['time', 'vehicle_mass', 'vehicle_propellant', 'pursuer_pos_x',
                        'pursuer_pos_y', 'pursuer_pos_z', 'pursuer_vel_x', 'pursuer_vel_y', 'pursuer_vel_z',
                        'evader_pos_x', 'evader_pos_y', 'evader_pos_z', 'evader_vel_x', 'evader_vel_y',
                        'evader_vel_z', "pursuer_orbital", "evader_orbital"]

            csv.writer(self.log_csv).writerow(head)

            log_name = log_name.replace("csv", "jsonl")
            self.log_jsonl = open(log_name, mode='w', newline='\n')


    def clean_response(self, response):
        # clean function args
        function_args = response

        # Eliminate "=> "
        function_args = function_args.replace('=>', '')

        # Find and extract perform action arguments.
        # Rather than searching for "perform_action(" we search for the first argument name
        # index = function_args.find("perform_action(")
        index = function_args.find('{"ft":')
        if index != -1:
            # Extract perform_action arguments
            # function_args = function_args[index + len("perform_action("):]
            function_args = function_args[index:]
            # index = function_args.find(")")
            index = function_args.find("}")
            if index != -1:
                function_args = function_args[:index+1]
            # Surround arguments with quotes
            function_args = function_args.replace("ft:", '"ft":')
            function_args = function_args.replace("rt:", '"rt":')
            function_args = function_args.replace("dt:", '"dt":')
            function_args = function_args.replace(',\n}', '\n}')
        function_args = function_args.replace("]", '}')
        function_args = function_args.replace("[", '{')

        index = function_args.find("Backward throttle(")
        if index != -1:
            print("Found Backward throttle")

        # Replace argument names used by CoT prompt
        #
        function_args = function_args.replace("Forward throttle", "ft")
        function_args = function_args.replace("Backward throttle", "ft")
        function_args = function_args.replace("Right throttle", "rt")
        function_args = function_args.replace("Up throttle", "dt")
        function_args = function_args.replace("Down throttle", "dt")

        # CoT prompt with speed limit returns the following ft values:
        # - full
        # - 0.5
        # We map them to "forward"
        #
        function_args = function_args.replace("full", "forward")
#        function_args = function_args.replace("0.5", "forward")

        # Now function arguments should be of the form:
        #   "{\n  \"ft\": 1,\n  \"rt\": -1,\n  \"dt\": 1\n}"
        #   "ft: 1, rt: -1, dt: 1"
        #   "ft -1 rt 1 dt 0"
        #   "-1, 0, 0"
        # Transform to the first form
        index = function_args.find("ft")
        if index == -1:
            # Case
            #   "1, 0 ,0"
            action = function_args.split(',')
            function_args = "{\n  \"ft\": " + action[0] + ",\n  \"rt\": " + action[1] + ",\n  \"dt\": " + action[2] + "\n}"
        else:
            index = function_args.find("{")
            if index == -1:
                # Cases:
                #   "ft: 1, rt: -1, dt: 1"
                #   "ft -1 rt 1 dt 0"

                # Add "{" and "}"
                function_args = "{" + function_args + "}"
                # Add colons
                index = function_args.find(":")
                if index == -1:
                    colons = ":"
                else:
                    colons = ""
                index = function_args.find(",")
                if index == -1:
                    comma = ","
                else:
                    comma = ""

                # Surround argument names with quotes
                function_args = function_args.replace("ft", '"ft"' + colons)
                function_args = function_args.replace("rt", comma + '"rt"' + colons)
                function_args = function_args.replace("dt", comma + '"dt"' + colons)

        if function_args[0] == "down":
            function_args[0] = "forward"

        # Find and extract next actions
        s = response
        index = s.find("Next predicted throttles are")
        pattern = r'[^\[]+([^\]]+)\]'
        s = s[index:]
        m = re.search(pattern, s)
        next_actions = []
        if m:
            try:
                data = json.loads(m.group(1) + ']')
                # Convert to list of actions
                for d in data:
                    action = Action.from_enum([d["ft"], d["rt"], d["dt"], self.duration])
                    next_actions.append(action)
            except Exception as ex:
                print("Exception processing next actions:" + s)

        return function_args, next_actions

    def check_response(self, response):
        if response is None:
            return None

        duration = self.duration
        available_functions = {
            "perform_action": lambda ft, rt, dt: [ft, rt, dt, duration],
        }
        next_actions = []
        if response.get("function_call"):
            function_name = response["function_call"]["name"]
            if function_name not in available_functions:
                print("error: LLM called wrong function, name:", function_name)
                function_response = [0, 0, 0, 0.1]
            else:
                function_to_call = available_functions[function_name]

                # Get function arguments
                function_args = response["function_call"]["arguments"]
                try:
                    function_args, next_actions = self.clean_response(function_args)
                    function_args = json.loads(function_args)
                    function_response = function_to_call(**function_args)
                    if self.use_enum:
                        function_response = Action.from_enum(function_response)
                except Exception as ex:
                    # traceback.print_exc()
                    function_response = [0, 0, 0, 0.1]
        else:
            function_name = "perform_action"
            function_to_call = available_functions[function_name]
            try:
                content = response["content"]
                # Assume that the response contains the perform_action call
                index = content.find(function_name + "(")
                index = 0
                if index != -1:
                    # Get function args from content
                    function_args, next_actions = self.clean_response(content)
                    function_args = json.loads(function_args)

                    # Add missing arguments
                    if "ft" not in function_args.keys():
                        function_args["ft"] = 0
                    if "rt" not in function_args.keys():
                        function_args["rt"] = 0
                    if "dt" not in function_args.keys():
                        function_args["dt"] = 0

                    # Eliminate extra arguments
                    for key in function_args.keys():
                        if key not in ["ft", "rt", "dt"]:
                            del function_args[key]

                    function_response = function_to_call(**function_args)
                    if self.use_enum:
                        if type(function_response[0]) == str:
                            function_response = Action.from_enum(function_response)
                else:
                    print("error: Response did not include call to perform_action")
                    function_response = [0, 0, 0, 0.1]
            except Exception as ex:
                # traceback.print_exc()
                function_response = [0, 0, 0, 0.1]

        return function_response, next_actions


    def get_completion(self, prompt, model="gpt-4"):
        """
        Send a prompt to the language model and return the response.
        """
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Add these print statements for better visibility
        # print(colored("\n=== SYSTEM PROMPT ===", "cyan"))
        # print(colored(self.system_prompt, "cyan"))
        # print(colored("\n=== USER PROMPT ===", "cyan"))
        # print(colored(prompt, "cyan"))
        # print(colored("\n=== FULL MESSAGES ===", "cyan"))
        # print(colored(json.dumps(messages, indent=2), "cyan"))

        tools = [{
            "type": "function",
            "function": {
                "name": "perform_action",
                "description": "Send the given throttles to the spacecraft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ft": {
                            "type": "string",
                            "enum": ["forward", "backward", "none"],
                            "description": "The forward throttle direction.",
                        },
                        "rt": {
                            "type": "string",
                            "enum": ["right", "left", "none"],
                            "description": "The right throttle direction.",
                        },
                        "dt": {
                            "type": "string",
                            "enum": ["up", "down", "none"],
                            "description": "The down throttle direction.",
                        },
                    },
                    "required": ["ft", "rt", "dt"],
                }
            }
        }]

        print("messages: " + str(messages))
        time_before = time.time()
        self.first_completion = False
        try:
            if self.use_llama:
                system_prompt = ""
                user_prompt = ""
                for item in messages:
                    if item['role'] == 'system':
                        system_prompt = item['content']
                    elif item['role'] == 'user':
                        user_prompt = messages[1]['content']
                response = self.client.post("/generate/", json={"system_prompt": system_prompt, "user_msg": user_prompt, "model_answer": ""})
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "perform_action"}}
                )
                response = response.choices[0].message
            status = "success"
            status_message = None

        except Exception as e:
            print ("Exception: " + str(e))
            status = "error"
            status_message = str(e)
            response = None
            result = None
            token_usage = {}
            status = "error"
            status_message = str(e)

        time_after = time.time()

        """ Log conversation
        """
        if self.log_jsonl is not None:
            log_entry = {
                "name": self.scenario,
                "kind": "llm",
                "status_code": status,
                "status_message": status_message,
                "start_time_ms": time_before * 1000,
                "end_time_ms": time_after * 1000,
                "inputs": str(messages[1]['content']),
                "outputs": str(response),
            }
            json.dump(log_entry, self.log_jsonl)
            self.log_jsonl.write('\n')
            self.log_jsonl.flush()

        print("Chat completion took: " + str(time_after - time_before) + " seconds")
        print("Response message: " + str(response))

        return response
    

    # Return sun position in the celestial body orbital reference frame
    def get_sun_position(self):
        reference_frame = self.body.orbital_reference_frame
        # Get the sun position in the given reference frame
        sun_pos = self.conn.space_center.bodies['Sun'].position(reference_frame)
        return sun_pos


    def get_action(self, observation, sun_position=None):
        """
        Generate actions based on observations using the loaded prompts.
        """
        print("get_action called, prompting ChatGPT...")

        print("Mission time: " + str(observation[0]))

        # Get vessel up direction in celestial body reference frame
        vessel_up = None
        vessel_up = self.conn.space_center.transform_direction((0, 0, 1),
                                                               self.vessel.reference_frame,
                                                               self.body.orbital_reference_frame)
        # BE CAREFUL
        vessel_up = State.lh_to_rh(vessel_up)

        # Get the sun position in the given reference frame
        if sun_position is None:
            sun_position = self.get_sun_position()

        # Build state and show it
        state = State(observation, vessel_up, sun_position)        
        state_json = state.to_json(
            self.scenario,
            use_relative_coordinates=self.use_relative_coordinates,
            use_short_names=self.use_short_argument_names,
            use_prograde=self.use_prograde,
            use_cot=self.use_cot,
            use_cot_speed_limit=self.use_cot_speed_limit
        )
        
        recommendation = Action(state.get_recommendation())
        recommendation_str = ', '.join(recommendation.to_enum())
        user_prompt = self.user_prompt_template.format(CoT=self.cot_prompt, obs=str(state_json), rec=recommendation_str)

        """ Log observations
        """
        if self.log_csv is not None:
            if len(observation) > 15:
                row = observation[0:21]
            else:
                row = observation[0:15]

            row = np.append(row, str(orbital_from_vectors(kerbin_mu, state.pursuer_position, state.pursuer_velocity)).replace("\n", ""))
            row = np.append(row, str(orbital_from_vectors(kerbin_mu, state.evader_position, state.evader_velocity)).replace("\n", ""))
            if len(observation) > 15:
                row = np.append(row, str(orbital_from_vectors(kerbin_mu, state.guard_position, state.guard_velocity)).replace("\n", ""))

            csv.writer(self.log_csv).writerow(row)
            self.log_csv.flush()

        try:
            if observation[0] <= 240:
                if self.use_llama:
                    # Need to replace spacecraft names to use the LlaMA models fine-tuned on the PE scenario
                    user_prompt = user_prompt.replace("bandit", "pursuer")
                    user_prompt = user_prompt.replace("lady", "evader")

                response = self.get_completion(prompt=user_prompt)
                if self.use_llama:
                    action, next_actions = self.check_response(response)
                else:
                    action_json = json.loads(response.tool_calls[0].function.arguments)
                    action = Action.from_enum([
                        action_json["ft"],
                        action_json["rt"],
                        action_json["dt"],
                        0.5  # Default duration
                    ])
            else:
                print(colored(f"No action after 120 seconds", "yellow"))
                action = [0, 0, 0, 0.5]

        except Exception as e:
            print(f"Exception: {e}")
            # traceback.print_exc()
            action = [0, 0, 0, 0.1]


        # Evasion tactis for LBG scenarios
        #
        if self.hack or self.scenario.lower().startswith("lbg"):
            # Calculate guard distance
            guard_position = np.array([observation[15], observation[16], observation[17]])
            guard_distance = np.linalg.norm(guard_position - state.pursuer_position, ord=2)
            print ("Guard distance: " + str(guard_distance))
            if GUARD_DISTANCE_LOW < guard_distance and guard_distance < GUARD_DISTANCE_HIGH:
                # Perform a guard evasive action
                print(colored(f"Guard evasive action", "yellow"))
                """
                action = [0, 1, 0, 0.5]
                """

        print(action)
        return {
            "burn_vec": action,
            "ref_frame": 0
        }


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process command-line arguments.")

    # Add the --log flag (optional)
    parser.add_argument('--log', help='Enable logging', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Load environment paths and prompts
    set_env_paths()
    prompts = load_prompts()

    # Get scenario from environment variable
    scenario = os.environ['SCENARIO']

    # Use the setup_scenarios utility from agent_common
    scenarios = setup_scenarios()

    if scenario not in scenarios:
        print(f"Invalid scenario: {scenario} not in {list(scenarios.keys())}")
        sys.exit(1)

    # Initialize and run the agent
    my_agent = SimpleLLMAgent(prompts, arg1=args.log)
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False
    )
    runner.run()

