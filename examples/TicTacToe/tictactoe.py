"""Check, train and watch the GAMA tic-tac-toe through GamaAECEnv.

    gama-headless.bat -socket 6868

    py tictactoe.py --check              # our GAMA port vs pettingzoo's tictactoe_v3
    py tictactoe.py --train              # tabular Q-learning, self-play
    py tictactoe.py --play --port 1000   # watch a game in the GAMA GUI

Tabular Q-learning on purpose: the state space is small enough that no network, no optimiser
and no hyperparameter search are needed. What is being demonstrated is that GamaAECEnv drives
a turn-based game correctly, and that is easier to believe when the learner is something
whose behaviour nobody doubts.

The scoreboard is absolute rather than relative: perfect tic-tac-toe is a draw, so a trained
player must never lose to a perfect opponent, and should beat a random one most of the time.
No hand-written heuristic of unknown quality is involved.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict

import nest_asyncio
import numpy as np
from importlib.metadata import version, PackageNotFoundError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gama_pettingzoo.gama_aec_env import GamaAECEnv  # noqa: E402

GAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "tictactoe.gaml"))
Q_PATH = os.path.join(os.path.dirname(__file__), "tictactoe_q.json")

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]



def _pkg_version(name):
    """gama_client n'expose pas de __version__ (son module racine est vide) : la version ne se
    lit que dans les metadonnees du paquet installe."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


def log_versions(port):
    print(f"gama_client {_pkg_version('gama_client')} | "
          f"gama_pettingzoo {_pkg_version('gama_pettingzoo')} | "
          f"pettingzoo {_pkg_version('pettingzoo')}", flush=True)
    # Le port est affiche ici parce que le defaut du script (6868) n'est pas forcement celui
    # du serveur lance : une erreur de connexion ressemble alors a une incompatibilite.
    print(f"GAMA attendu sur le port {port}", flush=True)


def make_env(port: int, render: bool = False) -> GamaAECEnv:
    """gama_client needs a *running* event loop while its socket is built."""
    nest_asyncio.apply()
    holder = {}

    async def _build():
        holder["env"] = GamaAECEnv(
            gaml_experiment_path=GAML,
            gaml_experiment_name="aec",
            gama_ip_address="localhost",
            gama_port=port,
            steps_per_round=0,          # the move IS the step; no clock to advance
            render_mode="human" if render else None,
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_build())
    return holder["env"]


def new_game(env):
    """Start a fresh game without reloading the experiment.

    `env.reset()` always reloads the GAMA experiment, which for a model that can clear its own
    state costs far more than the reset itself — noticeable when a run plays thousands of
    games. Worth a `reset(options=...)` fast path in the library one day; until then the
    example clears the board itself and re-primes the episode bookkeeping the way `reset()`
    does.
    """
    env.gama_client._execute_expression(env.experiment_id, "PetzAgent[0].new_game()")

    env.agents = list(env.possible_agents)
    env._agent_selector.reinit(env.agents)
    env.agent_selection = env._agent_selector.reset()
    env.rewards = {a: 0.0 for a in env.agents}
    env._cumulative_rewards = {a: 0.0 for a in env.agents}
    env.terminations = {a: False for a in env.agents}
    env.truncations = {a: False for a in env.agents}
    env.infos = {a: {} for a in env.agents}


def legal_from_obs(obs):
    """Empty cells, read straight off the observation planes (own marks, then opponent)."""
    occupied = np.asarray(obs[:9]) + np.asarray(obs[9:])
    return [i for i in range(9) if occupied[i] == 0]


def state_key(obs):
    """Board from the mover's point of view: 'o' own, 'x' opponent, '.' empty."""
    own, opp = obs[:9], obs[9:]
    return "".join("o" if own[i] else ("x" if opp[i] else ".") for i in range(9))


# ── opponents with known strength ────────────────────────────────────────────────

def random_move(obs, rng):
    return rng.choice(legal_from_obs(obs))


def _winner(cells):
    for a, b, c in LINES:
        if cells[a] != "." and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return None


def _minimax(cells, maximising):
    """Exact value of a position for the side to move ('o'). Small enough to solve outright."""
    # Terminal values are ALWAYS from 'o''s point of view, never flipped by whose turn it is:
    # a position 'o' has already won is worth +1 whoever is to move next. Flipping it here
    # made the "perfect" opponent play badly enough to lose 3 games in 50, which is what gave
    # the bug away — a solved game cannot be lost by optimal play.
    w = _winner(cells)
    if w == "o":
        return 1
    if w == "x":
        return -1
    empty = [i for i, c in enumerate(cells) if c == "."]
    if not empty:
        return 0

    scores = []
    for i in empty:
        nxt = cells.copy()
        nxt[i] = "o" if maximising else "x"
        scores.append(_minimax(nxt, not maximising))
    return max(scores) if maximising else min(scores)


def perfect_move(obs, rng):
    """Optimal play. A learner that never loses against this has genuinely solved the game."""
    cells = list(state_key(obs))
    best, best_score = [], -2
    for i in [i for i, c in enumerate(cells) if c == "."]:
        nxt = cells.copy()
        nxt[i] = "o"
        score = _minimax(nxt, False)
        if score > best_score:
            best, best_score = [i], score
        elif score == best_score:
            best.append(i)
    return rng.choice(best)


# ── tabular Q-learning ───────────────────────────────────────────────────────────

class QTable:
    def __init__(self, table=None):
        self.q = defaultdict(lambda: [0.0] * 9)
        if table:
            for k, v in table.items():
                self.q[k] = v

    def choose(self, obs, epsilon, rng):
        legal = legal_from_obs(obs)
        if epsilon and rng.random() < epsilon:
            return rng.choice(legal)
        values = self.q[state_key(obs)]
        return max(legal, key=lambda i: values[i])

    def best_value(self, obs):
        legal = legal_from_obs(obs)
        if not legal:
            return 0.0
        values = self.q[state_key(obs)]
        return max(values[i] for i in legal)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.q), f)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            return QTable(json.load(f))


def train(args):
    env = make_env(args.port)
    rng = random.Random(args.seed)
    q = QTable()
    t0 = time.time()
    try:
        env.reset()
        for game in range(args.games):
            epsilon = max(0.05, 1.0 - game / (0.7 * args.games))
            new_game(env)
            pending = {}                     # agent -> (state key, action) awaiting its reward

            for agent in env.agent_iter():
                obs, reward, termination, truncation, _ = env.last()

                if agent in pending:
                    key, action = pending.pop(agent)
                    target = reward if (termination or truncation) else \
                        reward + args.gamma * q.best_value(obs)
                    q.q[key][action] += args.lr * (target - q.q[key][action])

                if termination or truncation:
                    env.step(None)
                    continue

                action = q.choose(obs, epsilon, rng)
                pending[agent] = (state_key(obs), action)
                env.step(action)

            done = game + 1
            if done % args.progress_every == 0 and done % args.eval_every != 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[game {done:6d}] {rate:5.1f} parties/s | etats={len(q.q):5d} | "
                      f"reste ~{(args.games - done) / rate / 60:4.1f} min", flush=True)

            if (game + 1) % args.eval_every == 0:
                vs_rand = evaluate(env, q, random_move, args.eval_games, rng)
                vs_perf = evaluate(env, q, perfect_move, args.eval_games, rng)
                print(f"[game {game + 1:6d}] eps={epsilon:.2f} states={len(q.q):5d} | "
                      f"vs random  W{vs_rand['win']:3d} D{vs_rand['draw']:3d} L{vs_rand['loss']:3d}"
                      f" | vs perfect W{vs_perf['win']:3d} D{vs_perf['draw']:3d} "
                      f"L{vs_perf['loss']:3d}", flush=True)

        q.save(Q_PATH)
        print(f"\nQ-table saved -> {Q_PATH} ({len(q.q)} states)")
    finally:
        env.close()


def evaluate(env, q, opponent, games, rng):
    """Q-table plays player_1, `opponent` plays player_2."""
    score = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(games):
        new_game(env)
        final = 0.0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _ = env.last()
            if agent == "player_1":
                final = reward
            if termination or truncation:
                env.step(None)
                continue
            move = q.choose(obs, 0.0, rng) if agent == "player_1" else opponent(obs, rng)
            env.step(move)
        score["win" if final > 0 else ("loss" if final < 0 else "draw")] += 1
    return score


def check(args):
    """Replay identical move sequences in our GAMA port and in pettingzoo's tictactoe_v3."""
    from pettingzoo.classic import tictactoe_v3

    env = make_env(args.port)
    rng = random.Random(args.seed)
    ref = tictactoe_v3.env()
    mismatches = 0
    try:
        env.reset()
        for game in range(args.games):
            new_game(env)
            ref.reset(seed=game)

            while True:
                obs, _, term, trunc, _ = env.last()
                r_obs, _, r_term, r_trunc, _ = ref.last()

                if (term or trunc) != (r_term or r_trunc):
                    mismatches += 1
                    print(f"  game {game}: fin de partie differente "
                          f"(nous={term or trunc}, ref={r_term or r_trunc})")
                    break
                if term or trunc:
                    break

                ours_legal = legal_from_obs(obs)
                ref_legal = [i for i, m in enumerate(r_obs["action_mask"]) if m]
                if ours_legal != ref_legal:
                    mismatches += 1
                    print(f"  game {game}: coups legaux differents "
                          f"(nous={ours_legal}, ref={ref_legal})")
                    break

                move = rng.choice(ours_legal)
                env.step(move)
                ref.step(move)

                our_r = env.rewards["player_1"]
                ref_r = ref.rewards["player_1"]
                if abs(our_r - ref_r) > 1e-9:
                    mismatches += 1
                    print(f"  game {game}: recompense differente "
                          f"(nous={our_r}, ref={ref_r})")
                    break

        verdict = "IDENTIQUE" if mismatches == 0 else f"{mismatches} divergences"
        print(f"\n{args.games} parties comparees a tictactoe_v3 -> {verdict}")
    finally:
        env.close()
        ref.close()


def play(args):
    if not os.path.exists(Q_PATH):
        sys.exit(f"No Q-table at {Q_PATH} - train one first.")
    q = QTable.load(Q_PATH)
    rng = random.Random(args.seed)

    env = make_env(args.port, render=True)
    try:
        env.reset()
        new_game(env)
        for agent in env.agent_iter():
            obs, reward, termination, truncation, _ = env.last()
            if termination or truncation:
                print(f"  {agent:<9} reward {reward:+.0f}")
                env.step(None)
                continue
            move = q.choose(obs, 0.0, rng) if agent == "player_1" else perfect_move(obs, rng)
            print(f"  {agent:<9} plays {move}   board {state_key(obs)}")
            env.step(move)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=6868)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=8000)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--progress-every", type=int, default=200,
                   help="ligne de progression entre deux evaluations (0 pour desactiver)")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--check", action="store_true", help="compare against pettingzoo")
    p.add_argument("--play", action="store_true", help="watch one game (use with the GUI)")
    args = p.parse_args()

    log_versions(args.port)

    if args.check:
        args.games = min(args.games, 200)
        check(args)
    elif args.play:
        play(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
