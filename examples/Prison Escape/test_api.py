import asyncio
from pathlib import Path

import pettingzoo
from pettingzoo.test import parallel_api_test
from gama_pettingzoo.gama_parallel_env import GamaParallelEnv


async def main():
    
    exp_path = str(Path(__file__).parents[0] / "controler.gaml")
    exp_name = "main"

    env = GamaParallelEnv( gaml_experiment_path=exp_path,
                           gaml_experiment_name=exp_name,
                           gama_ip_address="localhost",
                           gama_port=1001)
    
    parallel_api_test(env, num_cycles=100)

    env.close()

if __name__ == "__main__":
    asyncio.run(main())