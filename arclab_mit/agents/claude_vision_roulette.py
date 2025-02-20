import argparse
import sys
import os
from kspdg.agent_api.runner import AgentEnvRunner
from arclab_mit.agents.vision_LLM_agent import VisionLLMAgent
from arclab_mit.agents.simple_LLM_agent import SimpleLLMAgent
import anthropic
import traceback
from arclab_mit.agents.agent_common import set_env_paths, setup_scenarios, Action, State
from termcolor import colored

class ClaudeVisionRoulette(VisionLLMAgent):
    def __init__(self, log=False, **kwargs):
        super().__init__(log=log, is_claude=True, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        self.model = "claude-3-5-sonnet-20241022"
        self.last_action = {"ft": "none", "rt": "none", "dt": "none"}
        self.simple_agent = SimpleLLMAgent()  # Create instance of SimpleLLMAgent for recommendations
        print(f"🤖 Using model: {self.model}")
        print(f"🔑 API Key present: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")

    def convert_recommendation_to_text(self, recommendation):
        """Convert numerical recommendation to text format"""
        # Get the values using to_enum() method
        ft, rt, dt, _ = recommendation.to_enum()

        print(f"Recommendation: {recommendation}")
        
        # No conversion needed since to_enum() already returns strings
        return f"ft: \"{ft}\", rt: \"{rt}\", dt: \"{dt}\""

    def get_completion(self, prompt, image=None, model=None):
        """Override to use Claude's API"""
        if model is None:
            model = self.model

        prograde = prompt['prograde']
        recommendation = prompt['recommendation']

        # Add debug prints
        print(colored("\n📝 Prompt content:", "cyan"))
        print("Prograde:", prograde)
        print("Recommendation:", recommendation)

        instruction_text = f"""You are an AI agent controlling a pursuit spacecraft tasked with capturing an evader spacecraft. Your goal is to manipulate like a pilot the spacecraft watching the KSP GUI as a dashboard.

        Here is the current prograde position:
        <prograde>
        {prograde}
        </prograde>

        I recommend to counteract the prograde and follow the prograde recommendation unless the guard is too close.

        IMPORTANT - A proven guidance system has provided this recommended movement:
        <recommendation>
        {recommendation}
        </recommendation>

        Only deviate from this recommendation if there is a need to evade the guard which is rounded with a purple color in the GUI."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<examples>\n<example>\n<PROGRADE>\n[-0.40179121974841087, -0.9149000699934227, 0.03900868696988538]\n</PROGRADE>\n<RECOMMENDATION>\nForward Throttle: forward, Right Throttle: right, Down Throttle: up\n</RECOMMENDATION>\n<ideal_output>..."  # rest of example
                    },
                    {
                        "type": "text",
                        "text": instruction_text
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
            print("✓ Image added to prompt")

        try:
            response = self.client.messages.create(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0,
                system=self.prompts["CLAUDE_VISION_SYSTEM_PROMPT"],
                tools=self.tools,
                tool_choice={"type": "tool", "name": "perform_action"}
            )
            
            print(colored("\n🤖 Claude's Response:", "green"))
            if response.content:
                for content in response.content:
                    print(content)
                    if content.type == 'tool_use':
                        print(colored("\n🎯 Tool Input:", "blue"))
                        print(content.input)
                        return content.input
            return None
        except Exception as e:
            print(colored(f"\n❌ Error getting completion: {e}", "red"))
            return None

    def get_action(self, observation):
        """Override to handle Claude's specific response structure"""
        print(colored("\n🤖 Vision-LLM Agent processing new observation...", "light_blue"))
        
        # Create State object to get prograde and recommendation
        vessel_up = self.conn.space_center.transform_direction(
            (0, 0, 1),
            self.vessel.reference_frame,
            self.body.orbital_reference_frame
        )
        vessel_up = State.lh_to_rh(vessel_up)
        
        # Create state object with vessel_up
        state = State(observation, vessel_up=vessel_up)
        prograde = state.get_prograde()
        
        # Get recommendation from state
        recommendation = Action(state.get_recommendation())
        recommendation_str = ', '.join(recommendation.to_enum())
        print(colored("\n💡 Recommendation:", "yellow"))
        print(recommendation_str)
        
        # Create prompt with both prograde and recommendation
        user_prompt = {
            'prograde': prograde,
            'recommendation': recommendation_str
        }
        
        # Get visual information
        print(colored("🎮 Getting visual input from KSP...", "light_blue"))
        image = self.get_visual_input()
        if image is not None:
            print(colored("✓ Visual input received", "green"))
        
        print(colored("💭 Prompting Claude with visual context...", "light_blue"))

        try:
            response = self.get_completion(prompt=user_prompt, image=image)
            
            if response:
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

        return {
            "burn_vec": action,
            "ref_frame": 0
        }

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    parser.add_argument('--log', help='Enable logging', action='store_true')
    args = parser.parse_args()

    # Load environment paths
    set_env_paths()

    # Get scenario from environment variable
    scenario = os.environ['SCENARIO']

    # Use the setup_scenarios utility from agent_common
    scenarios = setup_scenarios()

    if scenario not in scenarios:
        print(f"Invalid scenario: {scenario} not in {list(scenarios.keys())}")
        sys.exit(1)

    # Initialize and run the Claude vision roulette agent
    my_agent = ClaudeVisionRoulette(log=args.log)
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=200,
        debug=False
    )
    runner.run() 