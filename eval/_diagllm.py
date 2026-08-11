import sys; sys.path.insert(0,".")
import gymnasium as gym, numpy as np, travel_env.env_v2  # noqa
from travel_env.env_v2 import MAX_TURNS
from travel_env.llm import LLMClient, LLMError
from travel_env.llm_agent import LLMAgent
env = gym.make("travel_agent/TravelAgent-v2", max_episode_steps=MAX_TURNS)
obs, info = env.reset(seed=1)
llm = LLMClient(model="openai/gpt-5-mini"); llm.begin_episode()
ag = LLMAgent(llm)
# monkeypatch to surface the swallowed error
orig = llm.chat
def loud(*a, **k):
    try: return orig(*a, **k)
    except LLMError as e:
        print("LLMError:", str(e)[:600]); raise
llm.chat = loud
a = ag.act(obs, info, env.unwrapped)
print("action:", a, "call_failures:", ag.call_failures, "parse_failures:", ag.parse_failures)
