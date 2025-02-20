import numpy as np
from base64 import b64encode
import krpc
import PIL.Image
from io import BytesIO
import cv2
import os
import json
import time
from termcolor import colored
import sys
import tempfile
import os.path
import traceback
import logging
import requests
import csv
import datetime
import argparse

from arclab_mit.agents.agent_common import set_env_paths, setup_scenarios, Action, State, orbital_from_vectors
from arclab_mit.agents.simple_LLM_agent import SimpleLLMAgent, load_prompts

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env

# CONSTANTS
kerbin_mu = 3.5316000e12   # Gravitational parameter of Kerbin (m^3/s^2)
kerbin_radius = 600000     # Radius of Kerbin (m)
kerbin_mass = 5.2915158e22 # Mass of Kerbin (kg)

class VisionLLMAgent(SimpleLLMAgent):
   
    def __init__(self, prompts=None, log_file_prefix="vision_LLM_agent_log", **kwargs):
        # If prompts not provided, load them from alex_prompts.txt
        if prompts is None:
            prompts = load_prompts()  # This will load all prompts including LBG_SYSTEM_PROMPT_VISION
            
        super().__init__(prompts, log_file_prefix, **kwargs)
        
        # Vision-specific configurations
        self.max_image_size = 1920
        self.image_quality = 95
        self.screenshot_width = kwargs.get('screenshot_width', 1280)
        self.screenshot_height = kwargs.get('screenshot_height', 720)
        self.screenshot_width = kwargs.get('screenshot_width', 1280)
        self.screenshot_height = kwargs.get('screenshot_height', 720)

        # Only set GPT model if not using Claude
        if not kwargs.get('is_claude', False):
            self.model = "gpt-4o-2024-05-13"
            # self.model = "gpt-4o"

        # Enable logging for requests to print the request details
        # logging.basicConfig(level=logging.DEBUG)


    def prepare_image(self, image):
        """Prepare image for API submission"""
        try:
            # Resize if needed
            height, width = image.shape[:2]
            max_dim = max(height, width)
            if max_dim > self.max_image_size:
                scale = self.max_image_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Normalize and enhance image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
            # Adjust brightness and contrast
            alpha = 1.2  # Contrast
            beta = 0     # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Save a copy for debugging in C:\temp
            debug_path = r"C:\temp\debug_processed_image.jpg"
            cv2.imwrite(debug_path, image)
            print(colored(f"✓ Debug image saved to: {debug_path}", "green"))
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', image, [
                cv2.IMWRITE_JPEG_QUALITY, self.image_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            image_b64 = b64encode(buffer).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_b64}"
        
        except Exception as e:
            print(colored(f"❌ Error processing image: {e}", "red"))
            return None

    def get_completion(self, prompt, image=None, model=None):
        """Override to support image inputs"""
        if model is None:
            model = self.model

        messages = []
        if True:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add debug prints
        print(colored("\n=== SYSTEM PROMPT ===", "cyan"))
        print(colored(self.system_prompt, "cyan"))
        print(colored("\n=== USER PROMPT ===", "cyan"))
        print(colored(prompt, "cyan"))
        
        # Construct message content based on whether we have an image
        if image is not None:
            image_b64 = self.prepare_image(image)
            print(colored("\n=== IMAGE INFO ===", "cyan"))
            print(colored("Base64 string length: " + str(len(image_b64)), "yellow"))
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the best throttle to capture lady?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64,
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        print(colored("\n=== FULL MESSAGES ===", "cyan"))
        print(colored(json.dumps(messages, indent=2), "cyan"))
        
        # Use the same tools configuration as parent class
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
                        }
                    },
                    "required": ["ft", "rt", "dt"],
                }
            }
        }]

        # print("messages: " + str(messages))
        time_before = time.time()
        # print("Request message: " + str(messages[1]['content'][0]))
        try:
            if self.use_llama:
                if image is None:
                    response = self.client.post("/generate/", json={"system_prompt": self.system_prompt, "user_msg": prompt, "model_answer": ""})
                else:
                    # Uncomment the following block to debug the request details
                    """
                    # For debugging purposes
                    base_url = os.environ.get('LLAMA_API_URL', 'http://localhost:8000')
                    req = requests.Request('POST', base_url + '/generate_img/', json={"system_prompt": self.system_prompt, "user_msg": prompt, "image_url": image_b64, "model_answer": ""})
                    prepared_req = req.prepare()

                    # Print the request url
                    print(f"Request URL: {prepared_req.url}")
                    # Print the request headers
                    print(f"Request Headers: {prepared_req.headers}")
                    # Print the request body (form data in this case)
                    print(f"Request Body: {prepared_req.body[:1000]}")
                    print(f"Request Body: {prepared_req.body[-1000:]}")
                    """

                    response = self.client.post("/generate/", json={"system_prompt": self.system_prompt, "user_msg": prompt, "image_url": image_b64, "model_answer": ""})
    
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
                "inputs": str(messages[1]['content'][0]),
                "outputs": str(response),
            }
            json.dump(log_entry, self.log_jsonl)
            self.log_jsonl.write('\n')
            self.log_jsonl.flush()

        print("Chat completion took: " + str(time_after - time_before) + " seconds")
        print("Response message: " + str(response))
            
        return response

    def get_action(self, observation, sun_position=None):
        """Override to include visual information in decision making"""
        print(colored("\n🤖 Vision-LLM Agent processing new observation...", "light_blue"))
        
        # Get vessel up direction in celestial body reference frame
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

        # Get visual information
        print(colored("🎮 Getting visual input from KSP...", "light_blue"))
        image = self.get_visual_input()
        if image is not None:
            print(colored("✓ Visual input received", "green"))
        
        # Use the inherited prompt template
        recommendation = Action(state.get_recommendation())
        recommendation_str = ', '.join(recommendation.to_enum())
        user_prompt = self.user_prompt_template.format(CoT=self.cot_prompt, obs=str(state_json), rec=recommendation_str)
        print(colored("💭 Prompting LLM with visual context...", "light_blue"))

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
            response = self.get_completion(prompt=user_prompt, image=image)
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
            print(colored(f"✓ Action decided: {action}", "green"))
        except Exception as e:
            traceback.print_exc()
            print(colored(f"❌ Error processing action: {e}", "red"))
            action = [0, 0, 0, 0.1]

        return {
            "burn_vec": action,
            "ref_frame": 0
        }

    def get_visual_input(self):
        """Capture current game view from KSP using kRPC"""
        try:
            print(colored("📸 Capturing KSP screenshot...", "light_blue"))
            
            # Get the UI visible state and temporarily hide it if needed
            ui_was_visible = self.conn.ui.stock_canvas.visible
            self.conn.ui.stock_canvas.visible = False  # Hide UI for clean screenshot
            
            # Create a temporary file with absolute path
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "ksp_screenshot.jpg")
            
            print(colored(f"Saving to: {temp_file}", "light_blue"))
            
            # Take screenshot using the correct SpaceCenter method
            self.conn.space_center.screenshot(
                file_path=temp_file,
                scale=1
            )
            
            # Restore UI state
            self.conn.ui.stock_canvas.visible = ui_was_visible
            
            print(colored(f"✓ Screenshot captured", "green"))

            # Read the saved screenshot
            image_np = cv2.imread(temp_file)
            
            # Remove temporary file
            os.remove(temp_file)
            
            print(colored("✓ Image converted for processing", "green"))
            return image_np

        except Exception as e:
            print(colored(f"❌ Failed to capture screenshot: {e}", "red"))
            return None

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

    # Initialize and run the vision agent
    my_agent = VisionLLMAgent(prompts, arg1=args.log)
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=240,
        debug=False
    )
    runner.run()