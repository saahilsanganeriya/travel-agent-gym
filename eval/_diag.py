import sys; sys.path.insert(0,".")
import gymnasium as gym, collections
import travel_env.env_v2  # noqa
from travel_env.env_v2 import MAX_TURNS
from travel_env.actions_v2 import describe, decode
from eval.baselines_v2 import HeuristicPolicy
env = gym.make("travel_agent/TravelAgent-v2", max_episode_steps=MAX_TURNS)
obs, info = env.reset(seed=10001); pol = HeuristicPolicy(); pol.reset()
c = env.unwrapped; seq = collections.Counter(); acts=[]
for _ in range(MAX_TURNS):
    a = pol.act(obs, info, env); acts.append(describe(a)); seq[decode(a).verb.name]+=1
    obs, r, term, trunc, info = env.step(a)
    if term or trunc: break
print("verbs:", dict(seq))
print("last 12:", acts[-12:])
print(f"route={[(x[0].value,x[1]) for x in c.route]} nights={c.nights}")
print(f"out={c.timeline.outbound is not None} in={c.timeline.inbound is not None} stays={len(c.timeline.stays)} ground={len(c.timeline.ground)}")
print(f"pending_city={c._pending_city()} next_hop={c._next_hop()} complete={c.timeline.is_complete()}")
print(f"hotel_slate={len(c.hotel_slate)} committed={c._committed_cost()} budget={c.persona.budget_cents} headroom={c.persona.budget_cents-c._committed_cost()}")
