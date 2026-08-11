"""Load a trained checkpoint and play a rendered episode, then score it against baselines.

The point of committing a checkpoint is that someone can run it, not just see that it
exists. This is that script.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

import travel_env.env  # noqa: F401 — registers the env id
from travel_env.actions import describe
from travel_env.env import MAX_TURNS


def play(model_path: str, seed: int, episodes: int, render: bool) -> None:
    model = MaskablePPO.load(model_path, device="cpu")
    env = gym.make(
        "travel_agent/TravelAgent-v0", max_episode_steps=MAX_TURNS, render_mode="ansi"
    )
    returns, solved = [], 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total, done = 0.0, False
        if render and ep == 0:
            print(env.unwrapped.render())
            print("client:", env.unwrapped.transcript[0][1], "\n")
        while not done:
            # deterministic=True is argmax; the mask must be passed explicitly because
            # predict() does NOT fetch it for you.
            action, _ = model.predict(
                obs, deterministic=True, action_masks=info["action_mask"]
            )
            obs, reward, term, trunc, info = env.step(int(action))
            total += float(reward)
            done = term or trunc
            if render and ep == 0:
                print(f"  {describe(int(action)):<34} r={reward:+.4f}")
        returns.append(total)
        solved += int(info["itinerary_complete"] and not info["client_walked"] and total > 0)
        if render and ep == 0:
            print("\n" + env.unwrapped.render())

    print(
        f"\n{episodes} episodes  |  mean return {np.mean(returns):+.4f} "
        f"± {np.std(returns):.3f}  |  solve rate {solved/episodes:.0%}"
    )
    print("random baseline is ≈ -1.00 for comparison")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/v1_ppo/best_model")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    play(a.model, a.seed, a.episodes, not a.no_render)
