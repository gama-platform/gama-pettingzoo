"""Smoke test and demo for GamaAECEnv on the wildfire model.

Start a GAMA server, then::

    py run_wildfire.py                  # random policy
    py run_wildfire.py --policy nearest # move toward the nearest visible flame
    py run_wildfire.py --api-test       # PettingZoo API conformance test

`--policy nearest` is the one that shows why this model wants the AEC environment: each
firefighter heads for a flame it can see, and because a douse takes effect immediately, the
one playing later in the round already sees that flame gone and picks another. Run it in the
GAMA GUI to watch the fire and the team move.
"""

import argparse
import asyncio
import os
import sys

import nest_asyncio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gama_pettingzoo.gama_aec_env import GamaAECEnv  # noqa: E402

GAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "wildfire.gaml"))

# Observation layout, mirroring observe_firefighter in the GAML model.
BURNING = slice(0, 8)          # 8 neighbours, 1.0 if burning
POS = slice(16, 18)            # own grid position, normalised
FIRE_SHARE = 18                # share of the map alight
BEARING = slice(20, 22)        # offset to the nearest flame, centred on 0.5
STAY, NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3, 4


def make_env(port: int, steps_per_round: int) -> GamaAECEnv:
    """gama_client needs a *running* event loop while its socket is built; nest_asyncio lets
    us nest run_until_complete so synchronous code can do that."""
    nest_asyncio.apply()
    holder = {}

    async def _build():
        holder["env"] = GamaAECEnv(
            gaml_experiment_path=GAML,
            gaml_experiment_name="aec",
            gama_ip_address="localhost",
            gama_port=port,
            steps_per_round=steps_per_round,
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_build())
    return holder["env"]


def nearest_flame_action(obs: np.ndarray, rng) -> int:
    """Walk toward the nearest flame, using the bearing the model publishes.

    Deliberately greedy and uncoordinated: every firefighter heads for the closest fire it
    can see. That is exactly the policy the parallel API would make degenerate — all three
    would pick the same flame. Under AEC the douse lands before the next agent observes, so
    the later ones already see a different nearest fire.
    """
    dx, dy = obs[BEARING] - 0.5
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return int(rng.integers(0, 5))
    if abs(dx) >= abs(dy):
        return EAST if dx > 0 else WEST
    return SOUTH if dy > 0 else NORTH


def run_episode(env: GamaAECEnv, policy: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    print(f"agents: {env.possible_agents}")
    print(f"observation space: {env.observation_space(env.possible_agents[0])}")
    print(f"action space:      {env.action_space(env.possible_agents[0])}\n")

    totals = {a: 0.0 for a in env.possible_agents}
    round_index = 0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, _info = env.last()
        totals[agent] += reward

        if termination or truncation:
            env.step(None)
            continue

        if policy == "nearest":
            action = nearest_flame_action(obs, rng)
        else:
            action = int(env.action_space(agent).sample())

        # obs[16:18] is the agent's position, obs[18] the share of the map on fire. Both are
        # read *after* the earlier agents of this round have already acted.
        print(f"  round {round_index:>2} | {agent:<14} at ({obs[16]:.2f},{obs[17]:.2f}) "
              f"fire={obs[18]:.3f} flames_adjacent={int(obs[BURNING].sum())} -> move {action}")

        env.step(action)

        if env._agent_selector.is_first():
            print(f"  --- end of round {round_index}, rewards {env.rewards}\n")
            round_index += 1

    print("\nepisode over")
    for a, t in totals.items():
        print(f"  {a:<14} cumulative reward {t:>8.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=6868)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "nearest"], default="random")
    parser.add_argument("--steps-per-round", type=int, default=1,
                        help="GAMA cycles per round; 1 lets the spread reflex run")
    parser.add_argument("--api-test", action="store_true")
    args = parser.parse_args()

    env = make_env(args.port, args.steps_per_round)
    try:
        if args.api_test:
            from pettingzoo.test import api_test
            api_test(env, num_cycles=20, verbose_progress=True)
            print("API test passed")
        else:
            run_episode(env, args.policy, args.seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
