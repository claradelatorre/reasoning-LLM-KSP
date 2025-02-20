import argparse
import sys
import os
from kspdg.agent_api.runner import AgentEnvRunner
from arclab_mit.agents.vision_LLM_agent import VisionLLMAgent
import anthropic
import traceback
import json
from arclab_mit.agents.agent_common import set_env_paths, setup_scenarios, Action, State, orbital_from_vectors
from arclab_mit.agents.simple_LLM_agent import SimpleLLMAgent, load_prompts
from termcolor import colored
import time
import numpy as np
import csv

# CONSTANTS
kerbin_mu = 3.5316000e12   # Gravitational parameter of Kerbin (m^3/s^2)
kerbin_radius = 600000     # Radius of Kerbin (m)
kerbin_mass = 5.2915158e22 # Mass of Kerbin (kg)

USE_IMAGE = False

class ClaudeVisionAgent(VisionLLMAgent):
    def __init__(self, prompts=None, log_file_prefix="claude_LLM_agent_log", **kwargs):
        super().__init__(prompts, log_file_prefix=log_file_prefix, is_claude=True, **kwargs)

        # Process kwargs
        self.strategy = kwargs.get("arg2", 1)

        self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        # Claude models
        #
        # Performance from lowest to highest: Opus / Sonnet / Haiku
        # Intelligence from lowest to highest: Haiku / Sonnet / Opus
        #
        """
        self.model = "claude-3-haiku-20240307"
        self.model = "claude-3-5-haiku-20241022"
        self.model = "claude-3-sonnet-20240229"
        """
        self.model = "claude-3-5-sonnet-20241022"
        self.last_action = {"ft": "none", "rt": "none", "dt": "none"}
        print(f"🤖 Using model: {self.model}")
        print(f"🔑 API Key present: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")
        print("self.scenario: " + self.scenario)
        if self.scenario.startswith("LBG"):
            if self.strategy == 1:
                self.system_prompt = os.environ["LBG_SYSTEM_PROMPT_CLAUDE"]
                self.cot_prompt = os.environ["LBG_CHAIN_OF_THOUGHT_CLAUDE"]
                self.user_prompt_template = os.environ["LBG_USER_PROMPT_CLAUDE"]
            else:
                self.system_prompt = os.environ["LBG_SYSTEM_PROMPT_CLAUDE_II"]
                self.cot_prompt = os.environ["LBG_CHAIN_OF_THOUGHT_CLAUDE_II"]
                self.user_prompt_template = os.environ["LBG_USER_PROMPT_CLAUDE_II"]

    def process_response(self, content):
        available_functions = {
            "perform_action": lambda ft, rt, dt: [ft, rt, dt, 0.5],
        }
        function_name = "perform_action"
        function_to_call = available_functions[function_name]

        try:
            # Assume that the response contains the perform_action call
            index = content.find(function_name + "(")
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
                to_remove = []
                to_add = ["ft", "rt", "dt"]
                for key in function_args.keys():
                    if key not in to_add:
                        to_remove.append(key)
                for key in to_remove:
                    del function_args[key]

                function_response = function_to_call(**function_args)
            else:
                print("error: Response did not include call to perform_action")
                function_response = [0, 0, 0, 0.1]
                        
        except Exception as ex:
            traceback.print_exc()
            function_response = [0, 0, 0, 0.1]
        
        action_json = {"ft": function_response[0], "rt": function_response[1], "dt": function_response[2], "duration": function_response[3]}
        return action_json


    def get_completion(self, prompt, image=None, model=None):
        """Override to use Claude's API"""
        if model is None:
            model = self.model

        if USE_IMAGE:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
#                            "text": "<examples>\n<example>\n<PROGRADE>\n[-0.40179121974841087, -0.9149000699934227, 0.03900868696988538]\n</PROGRADE>\n<ideal_output>\n<analysis>\n1. Step-by-step breakdown of prograde coordinates:\n   x: 0.027811547560243455 (slightly positive)\n   y: 0.9979857174802993 (strongly positive)\n   z: 0.057017765017900165 (slightly positive)\n\n   This indicates that the evader is moving:\n   - Slightly to the right (x-axis)\n   - Strongly away from us (y-axis)\n   - Slightly upwards (z-axis)\n\n2. Planned throttle adjustments:\n   x-axis (rt): We need to move slightly left to counteract the rightward motion. Since the movement is small, we'll use a full left throttle.\n   y-axis (ft): The evader is moving strongly away from us, so we need to move forward at full throttle to close the distance.\n   z-axis (dt): The upward movement is slight, but we should counteract it with a downward thrust. However, since the x and y movements are more significant, we'll prioritize those and keep the z-axis neutral for now.\n\n3. Summary:\n   By applying full left throttle and full forward throttle, we'll be able to align ourselves with the evader's trajectory and close the distance. The slight upward movement of the evader is less significant compared to its forward and rightward motion, so we'll focus on those two axes for now.\n\n</analysis>\n\nBased on this analysis, the optimal throttle settings for interception are:\n\n<throttle_decision>\n{\n  \"ft\":forward ,\n  \"rt\": right,\n  \"dt\": up\n}\n</throttle_decision>\n</ideal_output>\n</example>\n</examples>\n\n"
                            "text":
                                "<examples>\n" + 
                                    "<example>\n<PROGRADE>\n[-0.40179121974841087, -0.9149000699934227, 0.33900868696988538]\n</PROGRADE>\n" +
                                    "<SPEED>40</SPEED>" +
                                    "<ideal_output>\n<analysis>\n"+
                                        "1. Step-by-step breakdown of prograde coordinates:\n   x: -0.4017912197484108755 (negative)\n   y: -0.914900069993422702993 (strongly negative)\n   z: 0.33900868696988538 (positive)\n\n" +
                                          "This indicates that the evader is moving:\n   - To the left (x-axis)\n   - Strongly towards us (y-axis)\n   - upwards (z-axis)\n\n" +
                                        "2. Planned throttle adjustments:\n   x-axis (rt): We need to move right to counteract the leftward motion.\n"+
                                          "   y-axis (ft): The evader is moving strongly towards us, but since the speed is greater than 30 so we do not need to apply throttle to avoid overshooting.\n"+
                                          "   z-axis (dt): We need to move down to counteract the upward motion.\n\n3. Summary:\n+"
                                          "   By applying full right and down throttle, we'll be able to align ourselves with the evader's trajectory and close the distance.\n\n</analysis>\n\n" +
                                          "   Based on this analysis, the optimal throttle settings for interception are:\n\n"+
                                          "   <throttle_decision>\n{\n  \"ft\":none ,\n  \"rt\": right,\n  \"dt\": down\n}\n</throttle_decision>\n"+
                                    "</ideal_output>\n" + 
                                "</example>\n" +
                                "<example>\n<PROGRADE>\n[-0.40179121974841087, -0.9149000699934227, -0.33900868696988538]\n</PROGRADE>\n" +
                                    "<SPEED>40</SPEED>" +
                                    "<ideal_output>\n<analysis>\n"+
                                        "1. Step-by-step breakdown of prograde coordinates:\n   x: -0.4017912197484108755 (negative)\n   y: -0.914900069993422702993 (strongly negative)\n   z: -0.33900868696988538 (positive)\n\n" +
                                          "This indicates that the evader is moving:\n   - To the left (x-axis)\n   - Strongly towards us (y-axis)\n   - downwards (z-axis)\n\n" +
                                        "2. Planned throttle adjustments:\n   x-axis (rt): We need to move right to counteract the leftward motion.\n"+
                                          "   y-axis (ft): The evader is moving strongly towards us, but since the speed is greater than 30 so we do not need to apply throttle to avoid overshooting.\n"+
                                          "   z-axis (dt): We need to move up to counteract the downward motion.\n\n3. Summary:\n+"
                                          "   By applying full right and up throttle, we'll be able to align ourselves with the evader's trajectory and close the distance.\n\n</analysis>\n\n" +
                                          "   Based on this analysis, the optimal throttle settings for interception are:\n\n"+
                                          "   <throttle_decision>\n{\n  \"ft\":none ,\n  \"rt\": right,\n  \"dt\": up\n}\n</throttle_decision>\n"+
                                    "</ideal_output>\n" + 
                                "</example>\n" +
                                "</examples>\n\n"
                        },
                        {
                            "type": "text",
                            "text": "You are an AI agent controlling a pursuit spacecraft tasked with capturing an evader spacecraft. Your goal is to determine the optimal throttle settings for interception based on the prograde position data.\n\nHere is the current prograde position:\n<prograde>\n{{PROGRADE}}\n</prograde>\n\nYour task is to analyze this information and determine the optimal throttle settings for interception. You must adhere to the following constraints:\n1. Throttles must be set in the vessel reference frame where:\n   - The x-axis points to the right.\n   - The y-axis points forward towards the target.\n   - The z-axis points upwards.\n2. Use directional control to effectively capture the evader.\n3. The goal is to counteract the evader's motion to facilitate interception.\n\nFollow these steps to complete your analysis:\n\n1. Analyze the prograde position to understand the evader's current trajectory.\n2. Based on the prograde coordinates, determine the necessary direction of throttles in x, y, and z axes to align with the evader's trajectory and close the distance.\n3. Convert your throttle decisions to discrete values: 1 (positive direction), 0 (no change), or -1 (negative direction).\n\nBefore providing your final decision, explain your reasoning inside <analysis> tags. Your explanation should include:\n1. A step-by-step breakdown of each prograde coordinate (x, y, z) and what it indicates about the evader's motion.\n2. Your planned throttle adjustments for each coordinate, explaining how they will counteract the evader's motion.\n3. A summary of how your combined throttle decisions will facilitate interception.\n\nAfter your analysis, provide your throttle decisions in the following format:\n\n<throttle_decision>\n{\n  \"ft\": [value for y-axis throttle],\n  \"rt\": [value for x-axis throttle],\n  \"dt\": [value for z-axis throttle]\n}\n</throttle_decision>\n\nRemember, your goal is to intercept the evader spacecraft as efficiently as possible. Always provide a detailed analysis for your decisions before giving the final throttle settings. Your analysis should consider the prograde marker to make the most effective decisions for interception."
                        }
                    ]
                }
            ]

            if image is not None:
                image_b64 = self.prepare_image(image)
                messages[0]["content"].append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64.split(",")[1]
                    }
                })

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

#        print ("system prompt:" + self.system_prompt)
#        print ("user prompt:" + str(messages))
        time_before = time.time()
        try:
            response = self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
                system=self.system_prompt,
                # tools=self.tools,
                # tool_choice={"type": "tool", "name": "perform_action"}
            )

            time_after = time.time()

            """ Add image for logging purposes only
            """
            if self.log_jsonl is not None:
                if image is not None:
                    image_b64 = self.prepare_image(image)
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64.split(",")[1]
                        }
                    })
#            print("response: " + str(response))
            print(colored("✓ Chat completion took: " + str(time_after - time_before) + " seconds", "green"))
            if response.content:
                for content in response.content:
                    if content.type == 'tool_use':
                        return content.input
                    elif content.type == 'text':
                        """ Log conversation
                        """
                        if self.log_jsonl is not None:
                            log_entry = {
                                "name": self.scenario,
                                "kind": "llm",
                                "status_code": "success",
                                "status_message": "",
                                "start_time_ms": time_before * 1000,
                                "end_time_ms": time_after * 1000,
                                "inputs": str(messages[0]['content']),
                                "outputs": str(content.text),
                            }
                            json.dump(log_entry, self.log_jsonl)
                            self.log_jsonl.write('\n')
                            self.log_jsonl.flush()
                        return self.process_response(content.text)
            return None
        except Exception as e:
            print(f"Error getting completion: {e}")
            return None


    def get_action(self, observation, sun_position=None):
        """Override to handle Claude's specific response structure"""
#        print(colored("\n🤖 Vision-LLM Agent processing new observation...", "light_blue"))
        print(colored("\n🤖 LLM Agent processing new observation...", "light_blue"))
        
        print(colored("📊 Mission time: ", "cyan") + str(round(observation[0], 2)))


        vessel_up = self.conn.space_center.transform_direction((0, 0, 1),
                                                               self.vessel.reference_frame,
                                                               self.body.orbital_reference_frame)
        vessel_up = State.lh_to_rh(vessel_up)

        # Create State object to get prograde
        state = State(observation, vessel_up=vessel_up)
        prograde = state.get_prograde()
        
        print(colored("📊 Prograde vector: ", "cyan") + str(prograde))
        
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
        
        # Calculate guard distance
        if self.scenario.lower().startswith("lbg"):
            guard_position = np.array([observation[15], observation[16], observation[17]])
            guard_distance = np.linalg.norm(guard_position - state.pursuer_position, ord=2)
            print(colored("📊 Guard distance: ", "cyan") + str(round(guard_distance, 2)))

        recommendation = Action(state.get_recommendation())
        recommendation_str = ', '.join(recommendation.to_enum())

        # Need to exchange y and z components of prograde for CLAUDE
        prograde = [prograde[0], prograde[2], prograde[1]]        
        prograde = [float(x) for x in prograde]

        if USE_IMAGE:
            user_prompt = f"Given guard distance {guard_distance}, what is the best throttle to capture the lady?"
        else:
            approaching_label = "approaching" if state.approaching else "receding"
            user_prompt = self.user_prompt_template.format(CoT=self.cot_prompt, mission_time=state.mission_time, prograde=prograde, guard_distance=guard_distance,
                                                           target_distance=state.distance, target_speed=state.velocity,
                                                           approaching_label=approaching_label)

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

        # Get visual information
        print(colored("🎮 Getting visual input from KSP...", "light_blue"))
        image = self.get_visual_input()
        if image is not None:
            print(colored("✓ Visual input received", "green"))
        
#        print(colored("💭 Prompting LLM with visual context...", "light_blue"))
        print(colored("💭 Prompting LLM without visual context...", "light_blue"))

        try:
            response = self.get_completion(prompt=user_prompt, image=image)
            if response:
                # Response is already a dictionary, no need for json.loads()
                action = Action.from_enum([
                    response["ft"],
                    response["rt"],
                    response["dt"],
                    0.5  # Default duration
                ])
                print(colored(f"✓ Action decided: {action}", "green"))
            else:
                raise Exception("No valid response from Claude")
        except Exception as e:
            traceback.print_exc()
            print(colored(f"❌ Error processing action: {e}", "red"))
            action = [0, 0, 0, 0.1]

#        input("Press Enter to continue...")
        return {
            "burn_vec": action,
            "ref_frame": 0
        }

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument('--log', help='Enable logging', action='store_true')
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

    # Initialize and run the Claude vision agent
    my_agent = ClaudeVisionAgent(prompts, arg1=args.log)
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=200,
        debug=False
    )
    runner.run()