from environment import TicTacToe
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import torch
#load env and players
env = TicTacToe(config={"render_mode": "human"})
player1 = RLModule.from_checkpoint("C:/Users/kouas/Documents/RL_revisions/tic_tac_toe/weights/learner_group/learner/rl_module/p1")
player2 = RLModule.from_checkpoint("C:/Users/kouas/Documents/RL_revisions/tic_tac_toe/weights/learner_group/learner/rl_module/p2")

#play game
#Simulate a game
obs, _ = env.reset()
terminated = {"__all__": False}
env.render()
for _ in range(100):
    while not terminated["__all__"]:
        if obs.get(np.str_('player1'), 0):
            #player 1's turn
            player_1_obs = obs.get(np.str_('player1'), 0)
            #convert to tensor
            player_1_obs["observation"] = torch.from_numpy(player_1_obs["observation"]).float()
            player_1_obs["action_mask"] = torch.from_numpy(player_1_obs["action_mask"]).float()
            action_distribution = player1._forward_inference(
                {"obs": player_1_obs}
            )
            action = action_distribution["action_dist_inputs"].argmax().item()
            obs, reward, terminated, truncated, info = env.step({"player1": action})
        else:
            #player 2's turn
            player_2_obs = obs.get(np.str_('player2'), 0)
            #convert to tensor
            player_2_obs["observation"] = torch.from_numpy(player_2_obs["observation"]).float()
            player_2_obs["action_mask"] = torch.from_numpy(player_2_obs["action_mask"]).float()
            action_distribution = player2._forward_inference(
                {"obs": player_2_obs}
            )
            action = action_distribution["action_dist_inputs"].argmax().item()
            obs, reward, terminated, truncated, info = env.step({"player2": action})
        env.render()
    obs, _ = env.reset()
    terminated = {"__all__": False}
