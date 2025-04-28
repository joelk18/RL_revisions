from environnement import Eight_american_env
from ray.rllib.core.rl_module.rl_module import RLModule
from gymnasium.spaces.utils import flatten_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
import numpy as np
import torch
#load env and players
env = Eight_american_env(config={"render_mode": "human"})
player1 = RLModule.from_checkpoint("C:/Users/kouas/Documents/RL_revisions/8_aremicains/weights/learner_group/learner/rl_module/p1")
player2 = RLModule.from_checkpoint("C:/Users/kouas/Documents/RL_revisions/8_aremicains/weights/learner_group/learner/rl_module/p2")
#play game
#Simulate a game
obs, _ = env.reset()
terminated = {"__all__": False}
env.render()
for _ in range(100):
    while not terminated["__all__"]:
        if obs.get(np.str_('player_1'), 0):
            #player 1's turn
            player_1_obs = obs.get(np.str_('player_1'), 0)
            #convert to tensor
            for key, value in player_1_obs["observation"].items():
                if isinstance(value, np.ndarray):
                    if key == "deposit":
                        player_1_obs["observation"][key] = torch.tensor([value])
                    else:
                        player_1_obs["observation"][key] = torch.from_numpy(value)
                else:
                    player_1_obs["observation"][key] = torch.tensor([value], dtype=torch.float32)
            player_1_obs["action_mask"] = torch.from_numpy(player_1_obs["action_mask"]).float()

            """
            player1_temporal_obs = player_1_obs["observation"]
            #convert to tensor each value of the dict
            for key, value in player1_temporal_obs.items():
                if isinstance(value, np.ndarray):
                    if key =="deposit":
                        player1_temporal_obs[key] = torch.tensor([value])
                    else:
                        player1_temporal_obs[key] = torch.from_numpy(value)
                else:
                    player1_temporal_obs[key] = torch.tensor([value], dtype=torch.float32)
            #print(player1_temporal_obs)
            #flatten the observation space
            player_1_obs["observation"] = flatten_inputs_to_1d_tensor(player1_temporal_obs, env.observation_spaces["player_1"]["observation"])
            #player_1_obs["observation"] = torch.from_numpy(player_1_obs["observation"]).float()
            player_1_obs["action_mask"] = torch.from_numpy(player_1_obs["action_mask"]).float()
            """
            action_distribution = player1._forward_inference(
                {"obs": player_1_obs}
            )
            action = action_distribution["action_dist_inputs"].argmax().item()
            obs, reward, terminated, truncated, info = env.step({"player_1": action})
        else:
            #player 2's turn
            player_2_obs = obs.get(np.str_('player_2'), 0)
            #convert to tensor
            for key, value in player_2_obs["observation"].items():
                if isinstance(value, np.ndarray):
                    if key == "deposit":
                        player_2_obs["observation"][key] = torch.tensor([value])
                    else:
                        player_2_obs["observation"][key] = torch.from_numpy(value)
                else:
                    player_2_obs["observation"][key] = torch.tensor([value], dtype=torch.float32)
            player_2_obs["action_mask"] = torch.from_numpy(player_2_obs["action_mask"]).float()
            """
            player2_temporal_obs = player_2_obs["observation"]
            #convert to tensor each value of the dict
            for key, value in player2_temporal_obs.items():
                if isinstance(value, np.ndarray):
                    if key =="deposit":
                        player2_temporal_obs[key] = torch.tensor([value])
                    else:
                        player2_temporal_obs[key] = torch.from_numpy(value)
                else:
                    player2_temporal_obs[key] = torch.tensor([value], dtype=torch.float32)
            #print(player2_temporal_obs)
            #flatten the observation space
            print(player2_temporal_obs)
            player_2_obs["observation"] = flatten_inputs_to_1d_tensor(player2_temporal_obs, env.observation_spaces["player_2"]["observation"])
            #player_2_obs["observation"] = torch.from_numpy(player_2_obs["observation"]).float()
            player_2_obs["action_mask"] = torch.from_numpy(player_2_obs["action_mask"]).float()
            """
            action_distribution = player2._forward_inference(
                {"obs": player_2_obs}
            )
            action = action_distribution["action_dist_inputs"].argmax().item()
            obs, reward, terminated, truncated, info = env.step({"player_2": action})
        env.render()
    obs, _ = env.reset()
    terminated = {"__all__": False}
