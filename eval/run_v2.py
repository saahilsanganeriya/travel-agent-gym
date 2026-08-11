import sys, numpy as np, gymnasium as gym
sys.path.insert(0, ".")
import travel_env.env_v2  # noqa
from travel_env.env_v2 import MAX_TURNS
from eval.baselines_v2 import ALL_POLICIES
from eval.harness import iqm, pass_hat_k

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = []
for name, cls in ALL_POLICIES.items():
    env = gym.make("travel_agent/TravelAgent-v2", max_episode_steps=MAX_TURNS)
    pol = cls()
    rets, solved, complete, walked, steps, elic, cons, cfail = [], [], [], [], [], [], [], []
    for ep in range(N):
        obs, info = env.reset(seed=10_000 + ep); pol.reset()
        total, n = 0.0, 0
        while True:
            a = pol.act(obs, info, env)
            obs, r, term, trunc, info = env.step(a); total += r; n += 1
            if term or trunc: break
        rets.append(total); steps.append(n)
        complete.append(float(info["itinerary_complete"])); walked.append(float(info["client_walked"]))
        elic.append(info["axes_elicited"]); cons.append(info["constraints_elicited"])
        cfail.append(float(info["constraint_failures"] > 0))
        solved.append(1 if (total > 0 and info["itinerary_complete"] and info["constraint_failures"] == 0) else 0)
    rows.append((name, np.mean(solved), iqm(rets), np.mean(complete), np.mean(walked),
                 np.mean(steps), np.mean(elic), np.mean(cons), np.mean(cfail),
                 pass_hat_k([solved[i:i+1] for i in range(len(solved))], 1) if False else np.mean(solved)))
    env.close()

print(f"{'policy':<17}{'solve':>7}{'ret_iqm':>10}{'complete':>10}{'walk':>7}{'steps':>7}{'prefs':>7}{'hard_q':>8}{'cfail':>7}")
for r in rows:
    print(f"{r[0]:<17}{r[1]:>7.2f}{r[2]:>10.4f}{r[3]:>10.2f}{r[4]:>7.2f}{r[5]:>7.1f}{r[6]:>7.2f}{r[7]:>8.2f}{r[8]:>7.2f}")
