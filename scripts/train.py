"""Train a policy on the travel environment with MaskablePPO.

This exists to answer the one question that decides whether the
environment is any good: **can an agent actually learn in here?** A reward you cannot train
against is a reward you have not validated — the failure mode is invisible by inspection,
and the reference custom-env implementation only found its own broken reward by watching
training fail to move.

MaskablePPO because the action space is `Discrete(63)` with state-dependent legality, and
masking beats penalising by a margin that is not close (Huang & Ontañón, arXiv:2006.14171).
`action_masks()` is the exact method name sb3-contrib looks up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import gymnasium as gym

import travel_env.env  # noqa: F401 — registers travel_agent/TravelAgent-v0
from travel_env.env import MAX_TURNS

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")


def make_env(**kwargs):
    """Build through `gym.make`, never by instantiating the class.

    `make_vec_env(TravelAgentEnv, ...)` constructs the class directly and therefore skips
    the registration — which means no `TimeLimit`, which means a policy that never reaches
    a terminal state loops forever. That is exactly what happened on the first run.
    """

    def _init():
        return Monitor(
            gym.make("travel_agent/TravelAgent-v0", max_episode_steps=MAX_TURNS, **kwargs)
        )

    return _init


def train(
    *,
    steps: int = 400_000,
    n_envs: int = 8,
    difficulty: float = 0.5,
    enable_act_two: bool = True,
    ent_coef: float = 0.02,
    run_name: str = "maskable_ppo",
) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    env_kwargs = {"difficulty": difficulty, "enable_act_two": enable_act_two}
    venv = SubprocVecEnv([make_env(**env_kwargs) for _ in range(n_envs)])
    eval_venv = SubprocVecEnv([make_env(**env_kwargs) for _ in range(2)])

    model = MaskablePPO(
        "MultiInputPolicy",  # the observation is a single-level Dict
        venv,
        verbose=1,
        device="cpu",  # SB3's own guidance: run PPO on CPU
        tensorboard_log=str(LOG_DIR),
        # Exploration matters here: the useful trajectory (interview -> search -> book) is
        # long, and a greedy policy collapses onto booking immediately without asking.
        ent_coef=ent_coef,
        n_steps=256,  # more frequent updates; env stepping dominates wall-clock
        batch_size=1024,
        learning_rate=3e-4,
        gamma=0.995,  # near-1: the reward is overwhelmingly terminal
        gae_lambda=0.95,
    )

    # eval_freq is PER-ENV in SB3, so the real interval is eval_freq * n_envs.
    eval_cb = MaskableEvalCallback(
        eval_venv,
        eval_freq=max(1, 20_000 // n_envs),
        n_eval_episodes=20,
        best_model_save_path=str(MODEL_DIR / run_name),
        log_path=str(LOG_DIR / run_name),
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=steps, callback=eval_cb, tb_log_name=run_name)
    model.save(MODEL_DIR / run_name / "final")
    print(f"saved -> {MODEL_DIR / run_name / 'final'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--no-act-two", action="store_true")
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--run-name", type=str, default="maskable_ppo")
    a = ap.parse_args()
    train(
        steps=a.steps, n_envs=a.n_envs, difficulty=a.difficulty,
        enable_act_two=not a.no_act_two, ent_coef=a.ent_coef, run_name=a.run_name,
    )
