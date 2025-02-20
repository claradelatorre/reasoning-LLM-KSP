from arclab_mit.agents.agent_common import set_env_paths, setup_scenarios, State, Action, orbital_from_vectors
from arclab_mit.agents.simple_LLM_agent import SimpleLLMAgent, load_prompts
from arclab_mit.agents.vision_LLM_agent import VisionLLMAgent

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.e1_envs import PE1_E1_I2_Env
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.pe1.e1_envs import PE1_E1_I4_Env
from kspdg.pe1.e3_envs import PE1_E3_I3_Env
import traceback
import json
import argparse
import os
import sys

class VisionRecommendationAgent(VisionLLMAgent):
    def __init__(self, prompts=None, log=False, **kwargs):
        super().__init__(prompts, log, **kwargs)
        
        # Modify the user prompt template to include recommendation
        self.user_prompt_template = """Based on the current state {obs}, 
I recommend the following action: {recommendation}

What do you think about this recommendation? Please analyze the visual information 
and either confirm this recommendation or suggest a different action if you think 
it would be more appropriate. Consider both the current state and what you can see 
in the image."""

    def get_action(self, observation):
        """Override to include recommendation in decision making"""
        print("\n🤖 Vision-Recommendation Agent processing new observation...")
        
        # Create State object and get recommendation
        state = State(observation, vessel_up=self.vessel.direction(self.vessel.orbital_reference_frame))
        recommendation = self.get_recommendation(state)
        
        # Convert recommendation to human-readable format
        recommendation_str = f"Forward: {'forward' if recommendation[0] > 0 else 'backward' if recommendation[0] < 0 else 'none'}, "
        recommendation_str += f"Right: {'right' if recommendation[1] > 0 else 'left' if recommendation[1] < 0 else 'none'}, "
        recommendation_str += f"Up: {'up' if recommendation[2] > 0 else 'down' if recommendation[2] < 0 else 'none'}"
        
        state_json = state.to_json(
            self.scenario,
            use_relative_coordinates=self.use_relative_coordinates,
            use_short_names=self.use_short_argument_names,
            use_prograde=self.use_prograde,
            use_cot=self.use_cot,
            use_cot_speed_limit=self.use_cot_speed_limit
        )

        # Get visual information
        print("🎮 Getting visual input from KSP...")
        image = self.get_visual_input()
        if image is not None:
            print("✓ Visual input received")
        
        # Use the modified prompt template
        user_prompt = self.user_prompt_template.format(
            CoT=self.cot_prompt, 
            obs=str(state_json),
            recommendation=recommendation_str
        )
        
        print("💭 Prompting LLM with visual context and recommendation...")

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
            print(f"✓ Action decided: {action}")
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Error processing action: {e}")
            action = [0, 0, 0, 0.1]

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

    # Initialize and run the vision agent
    my_agent = VisionLLMAgent(prompts, args.log)
    runner = AgentEnvRunner(
        agent=my_agent,
        env_cls=scenarios[scenario],
        env_kwargs=None,
        runner_timeout=100,
        debug=False
    )
    runner.run() 