import numpy as np
from typing import List, Dict
#from collections import deque
from scipy.stats import rv_discrete
class Environment():

    def __init__(self, policy_casino):
        self.cards: List[(int, float)] = [(card, 24/312) if card != 10 else (card, 96/312) for card in range(2, 12)]
        self.state: Dict = {"player_cards": [], "dealer_cards": []}
        self.custom_distribution: rv_discrete = rv_discrete(values=([element[0] for element in self.cards], [element[1] for element in self.cards]))
        self.policy_casino: int = policy_casino
        self.terminate: bool = False
    def compute_card_value(self, perso):
        if perso == "player":
            if (sum(self.state["player_cards"]) > 21) and (11 in self.state["player_cards"]):
                return sum(self.state["player_cards"])-10
            else:
                return sum(self.state["player_cards"])
        else:
            if (sum(self.state["dealer_cards"]) > 21) and (11 in self.state["dealer_cards"]):
                return sum(self.state["dealer_cards"])-10
            else:
                return sum(self.state["dealer_cards"])
    def compute_dealer(self):
        self.terminate = True
        while self.compute_card_value("dealer")<self.policy_casino:
            self.state["dealer_cards"].append(self.custom_distribution.rvs())
            #Comparison
            if (self.compute_card_value("dealer") > 21):
            #The player win
                return (self.compute_card_value("player"), self.state["dealer_cards"][0], 1, self.terminate)
        if self.compute_card_value("dealer") == self.compute_card_value("player"):
            #Equality
            return (self.compute_card_value("player"), self.state["dealer_cards"][0], 0, self.terminate)
        elif self.compute_card_value("dealer") > self.compute_card_value("player"):
            #Player lose
            return (self.compute_card_value("player"), self.state["dealer_cards"][0], -1, self.terminate)
        elif self.compute_card_value("dealer") < self.compute_card_value("player"):
            #Player win
            return (self.compute_card_value("player"), self.state["dealer_cards"][0], 1, self.terminate)

    def reset_state(self):
        self.state = {"player_cards": [], "dealer_cards": []}
        self.terminate = False
    def start(self):
        self.state["player_cards"]+= list(self.custom_distribution.rvs(size = 2))
        self.state["dealer_cards"]+= list(self.custom_distribution.rvs(size = 2))

        if self.compute_card_value("player") == 21 and self.compute_card_value("dealer") == 21: #draw
            self.terminate = True
            return (21, self.state["dealer_cards"][0], 0, self.terminate)
        elif self.compute_card_value("player") == 21: #Natural
            self.terminate = True
            return (21, self.state["dealer_cards"][0], 1, self.terminate)
        return (self.compute_card_value("player"), self.state["dealer_cards"][0], 0, self.terminate)
    
    def play(self, action):
        if action == 1: #hit
            self.state["player_cards"].append(self.custom_distribution.rvs())
            if self.compute_card_value("player") > 21:
                #We are above 21, it's a lost for the player
                self.terminate = True
                return (22, self.state["dealer_cards"][0], -1, self.terminate)
            elif self.compute_card_value("player") == 21:
                return self.compute_dealer()
            elif self.compute_card_value("player") < 21:
                return (self.compute_card_value("player"), self.state["dealer_cards"][0], 0, self.terminate)
                    
        if action == 0: #stick
            #Dealer getting his cards
            return self.compute_dealer()


#############
#### Agent ####
#############
class Agent():
    def __init__(self):
        self.value_action: np.ndarray = np.zeros((10, 18, 2))
        self.epsilon: float = 0.01
        self.alpha: float = 0.01
    
    def action(self, state):
        #epsilon greedy action selection
        if state[3]:
            return None
        if np.random.rand()>self.epsilon:
            return np.argmax(self.value_action[state[1]-2, state[0]-4, :])
        else:
            return np.random.randint(0, 2)
    def update_value_action_table(self, state, action, prev_state, prev_action):
        assert prev_state[0] < 21, f"We shouldn't update the state 21: actual state: {state}, previous state: {prev_state}"
        if state[3]:
            self.value_action[prev_state[1]-2, prev_state[0]-4, prev_action] = (self.alpha * (state[2]
                                                                            -self.value_action[prev_state[1]-2, prev_state[0]-4, prev_action]))
        else:
            self.value_action[prev_state[1]-2, prev_state[0]-4, prev_action] = (self.alpha * (state[2]+self.value_action[state[1]-2, state[0]-4, action]
                                                                            -self.value_action[prev_state[1]-2, prev_state[0]-4, prev_action]))
        