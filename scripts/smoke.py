"""End-to-end smoke: reset, random-but-legal rollout, terminal reward, reconciliation."""
import numpy as np, gymnasium as gym
import travel_env.env  # registers the id
from travel_env.actions import describe

env = gym.make("travel_agent/TravelAgent-v0", render_mode="ansi")
obs, info = env.reset(seed=7)
print("=== reset ok ===")
print(env.unwrapped.render())
print("opening:", env.unwrapped.transcript[0][1])
print("legal:", int(info["action_mask"].sum()), "/", len(info["action_mask"]))

rng = np.random.default_rng(0)
total, steps = 0.0, 0
term = trunc = False
while not (term or trunc):
    legal = np.flatnonzero(info["action_mask"])
    if len(legal) == 0:
        print("!! no legal actions at step", steps); break
    a = int(rng.choice(legal))
    obs, r, term, trunc, info = env.step(a)
    total += r; steps += 1
    if steps <= 5 or term:
        print(f"  {steps:>2} {describe(a):<34} r={r:+.4f} term={term}")
print(f"=== {steps} steps, return {total:+.4f}, term={term} trunc={trunc} ===")
print("complete:", info["itinerary_complete"], "| walked:", info["client_walked"],
      "| spent:", info["spent_cents"], "/", info["budget_cents"],
      "| elicited:", f"{info['axes_elicited']:.0%}")
print("reward parts:", {k: round(v, 4) for k, v in info.items() if k.startswith("reward_")})
