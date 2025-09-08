import asyncio
from pathlib import Path

import pettingzoo
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv


async def main():
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    exp_name = "main"

    env = GamaParallelEnv( gaml_experiment_path=exp_path,
                           gaml_experiment_name=exp_name,
                           gama_ip_address="localhost",
                           gama_port=1001)
    
    obs, infos = env.reset()
    print(f"Initial observations: {obs}")
    
    while env.agents:
    # for _ in range(5):  # Run for 5 steps
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, info = env.step(actions)
        print(f"Actions: {actions}, Obs: {obs}, Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncations}, Infos: {info}")

    env.close()
    
if __name__ == "__main__":
    asyncio.run(main())