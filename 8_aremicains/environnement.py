from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary, Discrete
import numpy as np
import random
from queue import Queue
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import List, Optional, Tuple
import pygame
from pygame.locals import QUIT

class Eight_american_env(MultiAgentEnv):
    def __init__(self, config: dict = None):
        super().__init__()
        self.nb_agents = config.get("nb_agents", 2)
        self.agents = self.possible_agents = ["player_1", "player_2"]
        self.nb_cards = config.get("nb_cards", 52)
        self.cards = [(color, number) for color in range(0, 4) for number in range(0, 13)]
        self.cards_queue: Queue = Queue()
        self.cards_to_number = {card: i for i, card in enumerate(self.cards)}
        self.number_to_card = {i: card for i, card in enumerate(self.cards)}
        self.temp_number_to_card = {52: (0, 7), 53: (0, 7), 54: (0, 7), 55: (0, 7),
                               56: (1, 7), 57: (1, 7), 58: (1, 7), 59: (1, 7),
                               60: (2, 7), 61: (2, 7), 62: (2, 7), 63: (2, 7),
                               64: (3, 7), 65: (3, 7), 66: (3, 7), 67: (3,7)}
        self.number_to_card.update(self.temp_number_to_card)
        self.number_to_color = {number: (i+4)%4 for (i, number) in enumerate(range(52, 68))}
        self.observation_spaces = Dict({
            "player_1": Dict({"observation": Dict({"cards": MultiBinary([4, 13]), "deposit": MultiDiscrete([4, 13]), "UNO": Discrete(2)}),
                        "action_mask": MultiBinary(69)}),
            "player_2": Dict({"observation": Dict({"cards": MultiBinary([4, 13]), "deposit": MultiDiscrete([4, 13]), "UNO": Discrete(2)}),
                        "action_mask": MultiBinary(69)}),
        })
        self.action_spaces = Dict({
            "player_1": Discrete(69),#[52-68] for spade, heart, diamond and green eight
            "player_2": Discrete(69),
        })
        self.deposit = None
        self.current_player = None
        self.keep_as_in_touch = 0
        self.keep_eight_in_touch = 0
        self.window = None
        self.clock = None
        #self.is_deck_empty = 0
    def check_uno(self, player: int):
        if np.sum(self.matrix_player[:, :, player-1]) == 1:
            return 1
        else:
            return 0
    
    def get_card_from_deck(self, number_cards: int, player: int) -> Optional[List]:
        #TODO: Shuffle the cards played already the put them in the deck
        cards = []
        if player:
            for _ in range(number_cards):
                if not self.cards_queue.empty():
                    card = self.cards_queue.get()
                    #Add to the matrix card player
                    self.matrix_player[card[0], card[1], player -1] = 1
                else:
                    #Fill the pit if deposit not empty
                    if len(self.deposit)>2:
                        temporal_cards = self.deposit[:-2].copy()
                        random.shuffle(temporal_cards)
                        #Fill the stack
                        for card in temporal_cards:
                            self.cards_queue.put(card)
                        #Deal to the player
                        card = self.cards_queue.get()
                        #Add to the matrix card player
                        self.matrix_player[card[0], card[1], player -1] = 1
                    else:
                        self.is_deck_empty = 1
                    #print("Deck empty")
            return None
        else:
            for _ in range(number_cards):
                if not self.cards_queue.empty():
                    cards.append(self.cards_queue.get())
                else:
                    print("Deck empty")
            return cards
    def get_action_mask(self, player: int, on_top_deposit: Tuple, action: Optional[int] = None) -> np.ndarray:
        action_mask = np.zeros((len(self.number_to_card)+1), dtype=np.int8)
        #As effect
        if action is None and on_top_deposit[1] == 1-1:
            self.keep_as_in_touch = 1
        elif action is not None and action == len(self.number_to_card):
            self.keep_as_in_touch = 0
        elif action is not None and self.number_to_card[action][1] == 1-1 and self.keep_as_in_touch == 0:# and self.deposit[-2][1] != 1:
            self.keep_as_in_touch = 1
        else:
            self.keep_as_in_touch = 0
        
        #Eight effect
        if action is not None and (action > 51 and action < len(self.number_to_card)):
            self.keep_eight_in_touch = action
        elif action is None and self.deposit[-1][1] == 8-1:
            self.keep_eight_in_touch = random.randint(52, len(self.number_to_card)-1)

        #As played
        if on_top_deposit[1] == 1-1 and self.keep_as_in_touch:
            for num_card in range(52):
                if self.matrix_player[self.number_to_card[num_card][0], self.number_to_card[num_card][1], player-1] == 1 and (self.number_to_card[num_card][1] == on_top_deposit[1]):
                     action_mask[num_card] = 1
        #Eight played
        elif on_top_deposit[1] == 8-1:
            for num_card in range(52):
                if self.matrix_player[self.number_to_card[num_card][0], self.number_to_card[num_card][1], player-1] == 1 and self.number_to_card[num_card][0] == self.number_to_color[self.keep_eight_in_touch]:
                    action_mask[num_card] = 1
            for num_card in range(52, len(self.number_to_card)):
                if self.matrix_player[self.number_to_card[num_card][0], self.number_to_card[num_card][1], player-1] == 1:
                    action_mask[num_card] = 1
        else:
            for num_card in range(52):
                if self.matrix_player[self.number_to_card[num_card][0], self.number_to_card[num_card][1], player-1] == 1 and (self.number_to_card[num_card][0] == on_top_deposit[0] or self.number_to_card[num_card][1] == on_top_deposit[1]):
                    action_mask[num_card] = 1
            for num_card in range(52, len(self.number_to_card)):
                if self.matrix_player[self.number_to_card[num_card][0], self.number_to_card[num_card][1], player-1] == 1:
                    action_mask[num_card] = 1
        #Lock the direct eight
        for i in range(0, 4):
            action_mask[self.cards_to_number[(i, 8-1)]] = 0
        #It is forbidden to draw when you have no more cards in the fake and in the pack
        if (len(self.deposit[:-1]) < 5 and self.cards_queue.qsize()< 5):
            action_mask[-1] = 0
        else:
            action_mask[-1] = 1
        return action_mask
    
    def reset(self, *, seed=None, options=None):
        dealer = random.randint(1, 2)
        self.matrix_player = np.zeros((4, 13, 2), dtype=np.int8)
        self.cards = [(color, number) for color in range(0, 4) for number in range(0, 13)]
        random.shuffle(self.cards)
        #Fill the stack
        for card in self.cards:
            self.cards_queue.put(card)
        #Initialize the state of each player
        if dealer == 1:
            self.current_player = 2
            self.opponent = 1
            self.get_card_from_deck(5, self.current_player)
            self.get_card_from_deck(5, self.opponent)
            self.deposit = self.get_card_from_deck(1, 0)
        else:
            self.current_player = 1
            self.opponent = 2
            self.get_card_from_deck(5, self.current_player)
            self.get_card_from_deck(5, self.opponent)
            self.deposit = self.get_card_from_deck(1, 0)
        #self.current_player = 1 if dealer ==2 else 2
        #Check if the card from deposit have a speciality
        if self.deposit[-1][1] == 10-1 or self.deposit[-1][1] == 11-1:
            self.current_player = 2 if dealer == 1 else 1
        #elif self.deposit[-1][1] == 8-1:
        #    self.deposit += self.get_card_from_deck(1, 0)
            #TODO: Add the fact that we can deal with only play a card by the color said by the opponent
        elif self.deposit[-1][1] == 2-1:
            self.get_card_from_deck(3, self.current_player)
            self.current_player = 2 if dealer == 1 else 1
        #Check if AS in top
        #elif self.deposit[-1][1] == 1-1:
        #    self.keep_as_in_touch = 1
        #Create the action mask
        action_mask = self.get_action_mask(self.current_player, self.deposit[-1])
        #Return the observation for the next player
        return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                }
                    },
                {}) #<- empty info dict)
        
    def step(self, action_dict):
        action = action_dict[f"player_{self.current_player}"]
        rewards = {f"player_{self.current_player}": 0.0}
        #if True, the episode ends for all agents.
        terminateds = {"__all__": False}
        if action == len(self.number_to_card):
            #The player want a new card
            self.get_card_from_deck(1, self.current_player)
            self.current_player = self.opponent
            self.opponent = 1 if self.current_player == 2 else 2
            action_mask = self.get_action_mask(self.current_player, self.deposit[-1], action)
            return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                    }
                },
            rewards,
            terminateds,
            {},
            {}) #<- empty info dict
        #Check if we have a valid action
        assert self.matrix_player[self.number_to_card[action][0], self.number_to_card[action][1], self.current_player-1] == 1, f"{self.matrix_player[:, :, self.current_player-1]}, Card played by player {self.current_player} isn't in his deck | card played: {self.number_to_card[action]}, the deposit: {self.deposit[-2]}"
        #Add the card played on the deposit
        self.deposit.append(self.number_to_card[action])
        #Remove the card from the player deck
        self.matrix_player[self.number_to_card[action][0], self.number_to_card[action][1], self.current_player-1] = 0

        if self.number_to_card[action][1] == 10-1 or self.number_to_card[action][1] == 11-1 or self.number_to_card[action] == 2-1:
            if self.number_to_card[action] == 2-1:
                self.get_card_from_deck(3, self.opponent)
            if (1 not in self.matrix_player[:, :, self.current_player-1]):
                self.get_card_from_deck(1, self.current_player)
            #The current player don't change
            action_mask = self.get_action_mask(self.current_player, self.deposit[-1], action)
            return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                    }
                },
            rewards,
            terminateds,
            {},
            {}) #<- empty info dict

        #End the game if there is no more cards in the pill and the fosse
        elif len(self.deposit[:-1]) == 0 and self.cards_queue.empty():
            terminateds = {"__all__": True}
            rewards[f"player_{self.opponent}"] = 0.0
            self.current_player = self.opponent
            self.opponent = 1 if self.current_player == 2 else 2
            action_mask = self.get_action_mask(self.current_player, self.deposit[-1])
            return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                    }
                },
            rewards,
            terminateds,
            {},
            {}) #<- empty info dict
        
        else:
            #Check validity for different cases from the previous card
            #For As
            if self.deposit[-2][1] == 1-1 and self.keep_as_in_touch:
                assert (self.number_to_card[action][1] == 1-1), f"Illegal move by player {self.current_player}, You should play an As"
            #For Eight
            elif self.deposit[-2][1] == 8-1:
                assert self.number_to_card[action][0] == self.number_to_color[self.keep_eight_in_touch] or self.number_to_card[action][1] == 8-1, f"Illegal move by player {self.current_player}, common It's an eight | card played: {self.number_to_card[action]}, the deposit: {self.deposit[-2]}"
            elif self.deposit[-1][1] == 8-1:
                #ok it all right
                assert 0 == 0, "My bad"
            else:
                assert self.number_to_card[action][0] == self.deposit[-2][0] or self.number_to_card[action][1] == self.deposit[-2][1], f"Bad move, card played: {self.number_to_card[action]}, the deposit: {self.deposit[-2]}"
            #Check if current player out of card
            if (1 not in self.matrix_player[:, :, self.current_player-1]):
                terminateds = {"__all__": True}
                rewards[f"player_{self.current_player}"] = 1
                rewards[f"player_{self.opponent}"] = -1
                print("The winner is: ", self.current_player)
                self.current_player = self.opponent
                self.opponent = 1 if self.current_player == 2 else 2
                action_mask = self.get_action_mask(self.current_player, self.deposit[-1])
                return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                    }
                },
            rewards,
            terminateds,
            {},
            {}) #<- empty info dict
            
            else:
                self.current_player = self.opponent
                self.opponent = 1 if self.current_player == 2 else 2
                action_mask = self.get_action_mask(self.current_player, self.deposit[-1])
                return ({f"player_{self.current_player}": 
                    {"observation":
                        {"cards": self.matrix_player[:,:, self.current_player-1], "deposit": np.array(self.deposit[-1], dtype=np.int64), "UNO": self.check_uno(self.opponent)},
                    "action_mask": action_mask
                    }
                },
            rewards,
            terminateds,
            {},
            {}) #<- empty info dict
            
    def render(self, mode="human"):
        #Colors
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        #Load card images (placeholder rectangles for now)
        def draw_card(surface, card, x, y):
            #color = red if card[0] == 0 else black
            match card[0]:
                case 0: color = red
                case 1: color = black
                case 2: color = blue
                case 3: color = green
            pygame.draw.rect(surface, white, (x, y, self.card_width, self.card_height))
            pygame.draw.rect(surface, color, (x, y, self.card_width, self.card_height), 2)
            font = pygame.font.Font(None, 24)
            text = font.render(f"{card[1] + 1}", True, color)
            surface.blit(text, (x + 10, y + 10))
        try:
            if self.window is None:
                # Initialize pygame
                pygame.init()
                # Screen dimensions
                self.screen_width = 800
                self.screen_height = 600
                self.card_width = 60
                self.card_height = 90
                self.margin = 20
                self.window = True
                # Create screen
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self.clock = pygame.time.Clock()
                pygame.display.set_caption("Eight American Card Game")

            # Main render loop
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    self.window = None
                    return
            # Clear screen
            self.screen.fill((0, 255, 0))

            # Draw deposit pile
            if self.deposit:
                draw_card(self.screen, self.deposit[-1], self.screen_width // 2 - self.card_width // 2, self.screen_height // 2 - self.card_height // 2)

            # Draw player cards
            for i, player in enumerate(["player_1", "player_2"]):
                y_offset = self.screen_height - self.card_height - self.margin if player == "player_1" else self.margin
                x_offset = self.margin
                for nb_card in range(52):
                    card = self.number_to_card[nb_card]
                    if self.matrix_player[card[0], card[1], i] == 1:
                        draw_card(self.screen, card, x_offset, y_offset)
                        x_offset += self.card_width + self.margin

            # Update display
            pygame.display.flip()
            self.clock.tick(0.1)  # Limit to 1 FPS
            
        except Exception as e:
            print(f"An error occurred: {e}")