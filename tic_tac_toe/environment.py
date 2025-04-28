import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict
import pygame
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TicTacToe(MultiAgentEnv):
    """A two-player game in which any player tries to complete one row in a 3x3 field.

    The observation space is Box(0.0, 1.0, (9,)), where each index represents a distinct
    field on a 3x3 board and values of 0.0 mean the field is empty, -1.0 means
    the opponend owns the field, and 1.0 means we occupy the field:
    ----------
    | 0| 1| 2|
    ----------
    | 3| 4| 5|
    ----------
    | 6| 7| 8|
    ----------

    The action space is Discrete(9) and actions landing on an already occupied field
    are simply ignored (and thus useless to the player taking these actions).

    Once a player completes a row, they receive +1.0 reward, the losing player receives
    -1.0 reward. In all other cases, both players receive 0.0 reward.
    """
    def __init__(self, config=None):
        super().__init__()

        # Define the agents in the game.
        self.agents = self.possible_agents = ["player1", "player2"]

        # Each agent observes a 9D tensor, representing the 3x3 fields of the board.
        # A 0 means an empty field, a 1 represents a piece of player 1, a -1 a piece of
        # player 2.
        self.observation_spaces = Dict({
            "player1": Dict({"observation": gym.spaces.Box(-1.0, 1.0, (9,), np.float32),
                             "action_mask": gym.spaces.Box(0, 1, (9,), np.float32)}),
            "player2": Dict({"observation": gym.spaces.Box(-1.0, 1.0, (9,), np.float32),
                             "action_mask": gym.spaces.Box(0, 1, (9,), np.float32)})
        })
        # Each player has 9 actions, encoding the 9 fields each player can place a piece
        # on during their turn.
        self.action_spaces = {
            "player1": gym.spaces.Discrete(9),
            "player2": gym.spaces.Discrete(9),
        }

        self.board = None
        self.current_player = None
        self.render_mode = config.get("render_mode", None)
        self.window = None
        self.clock = None
        self.mask_actions = np.ones((9,), dtype=np.float32)
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
    def reset(self, *, seed=None, options=None):
        self.board = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        # Pick a random player to start the game.
        self.mask_actions = np.ones((9,), dtype=np.float32)
        self.current_player = np.random.choice(["player1", "player2"])
        # Render the initial state
        #self.render(mode= self.render_mode)
        # Return observations dict (only with the starting player, which is the one
        # we expect to act next).
        return {
            self.current_player: {"observation": np.array(self.board, np.float32),
                                  "action_mask": self.mask_actions},
        }, {}

    def step(self, action_dict):
        action = action_dict[self.current_player]

        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {self.current_player: 0.0}
        # Create a terminateds-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminateds = {"__all__": False}

        opponent = "player1" if self.current_player == "player2" else "player2"

        # Penalize trying to place a piece on an already occupied field| check if agent move is legal
        assert self.board[action] == 0, f"illegal move from player {self.current_player[-1]} at position {action}"
        """
        if self.board[action] != 0:
            rewards[self.current_player] -= 5.0
        """
        # Change the board according to the (valid) action taken.
        self.board[action] = 1 if self.current_player == "player1" else -1
        #Mask_action
        self.mask_actions[action] = 0
        if 1 not in self.mask_actions:
            self.mask_actions = np.ones((9,), dtype=np.float32)
        # After having placed a new piece, figure out whether the current player
        # won or not.
        if self.current_player == "player1":
            win_val = [1, 1, 1]
        else:
            win_val = [-1, -1, -1]
        if (
            # Horizontal win.
            self.board[:3] == win_val
            or self.board[3:6] == win_val
            or self.board[6:] == win_val
            # Vertical win.
            or self.board[0:7:3] == win_val
            or self.board[1:8:3] == win_val
            or self.board[2:9:3] == win_val
            # Diagonal win.
            or self.board[::4] == win_val
            or self.board[2:7:2] == win_val
        ):
            # Final reward is +5 for victory and -5 for a loss.
            rewards[self.current_player] += 5.0
            rewards[opponent] = -5.0
            print("the winner is: ", self.current_player)
            # Episode is done and needs to be reset for a new game.
            terminateds["__all__"] = True

            # The board might also be full w/o any player having won/lost.
            # In this case, we simply end the episode and none of the players receives
            # +1 or -1 reward.
        elif 0 not in self.board:
            terminateds["__all__"] = True

        # Flip players and return an observations dict with only the next player to
        # make a move in it.
        self.current_player = opponent

        # Render the current state after the move
        #self.render(mode= self.render_mode)

        return (
            {self.current_player: {"observation": np.array(self.board, np.float32),
                                   "action_mask": self.mask_actions}
             },
            rewards,
            terminateds,
            {},
            {},
        )
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "matrix":
            return self.render_matrix()
        else:
            return None
        
    def render_matrix(self):
        # Convert board values to symbols
        symbols = {0: " ", 1: "X", -1: "O"}
        board_str = [symbols[val] for val in self.board]
        
        # Print the board in a 3x3 grid format
        print(f"\n{board_str[0]} | {board_str[1]} | {board_str[2]}")
        print("---------")
        print(f"{board_str[3]} | {board_str[4]} | {board_str[5]}")
        print("---------") 
        print(f"{board_str[6]} | {board_str[7]} | {board_str[8]}\n")

    def _render_frame(self):
        try:
            # Initialize pygame and window if not already done
            if self.window is None and self.render_mode == "human":
                pygame.init()
                self.screen_size = 600
                self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
                self.clock = pygame.time.Clock()
                pygame.display.set_caption('Tic Tac Toe')
                self.window = True

            # Safety check
            if not self.render_mode == "human" or not self.window:
                return None

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.window = None
                    return None

            # Draw board
            self.screen.fill(self.WHITE)
            
            # Draw grid lines
            line_width = 2
            cell_size = self.screen_size // 3
            
            # Draw horizontal and vertical lines
            for i in range(1, 3):
                # Vertical lines
                pygame.draw.line(
                    self.screen, 
                    self.BLACK,
                    (i * cell_size, 0),
                    (i * cell_size, self.screen_size),
                    line_width
                )
                # Horizontal lines
                pygame.draw.line(
                    self.screen,
                    self.BLACK,
                    (0, i * cell_size),
                    (self.screen_size, i * cell_size),
                    line_width
                )

            # Draw X's and O's
            for i in range(9):
                row = i // 3
                col = i % 3
                center_x = col * cell_size + cell_size // 2
                center_y = row * cell_size + cell_size // 2
                
                if self.board[i] == 1:  # X
                    offset = cell_size // 4
                    pygame.draw.line(
                        self.screen,
                        self.BLACK,
                        (center_x - offset, center_y - offset),
                        (center_x + offset, center_y + offset),
                        line_width
                    )
                    pygame.draw.line(
                        self.screen,
                        self.BLACK,
                        (center_x + offset, center_y - offset),
                        (center_x - offset, center_y + offset),
                        line_width
                    )
                elif self.board[i] == -1:  # O
                    radius = cell_size // 4
                    pygame.draw.circle(
                        self.screen,
                        self.BLACK,
                        (center_x, center_y),
                        radius,
                        line_width
                    )

            # Update display
            pygame.display.flip()
            
            # Control frame rate
            if self.clock is not None:
                self.clock.tick(1)  # Limit to 1 FPS
                
        except Exception as e:
            print(f"Render error: {e}")
            pygame.quit()
            self.window = None
            return None


if __name__ == "__main__":
    env = TicTacToe()
    obs, info = env.reset()
    terminated = False
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        clock.tick(30)  # Limit to 30 FPS
        
        # Handle pygame events in the main loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                exit()
        
        if not terminated:
            action = env.action_spaces["player1"].sample()  # Random action
            if obs.get("player1") is not None:
                obs, reward, terminated, truncated, info = env.step({"player1": action})
            else:
                action = env.action_spaces["player2"].sample()  # Random action for player 2
                obs, reward, terminated, truncated, info = env.step({"player2": action})
            
            pygame.time.wait(500)  # Add delay between moves to make it visible
        
        if terminated:
            pygame.time.wait(1000)  # Wait a second before resetting
            obs, info = env.reset()
            terminated = False
