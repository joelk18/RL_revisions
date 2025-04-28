import numpy as np
import matplotlib.pyplot as plt
"""
S0 = np.random.randint(1, 10)

#state return function
def state_reward(action, S):

    #Coin launch
    if np.random.randint(2, size=1)[0] == 1:
        state = S + action*2
        if state >= 100:
            reward = 1
    else:
        state = S - action
        reward = 0
    return (state, reward)

#Agent definition
class Agent():
    def __init__(self):
        value_function = np.zeros(100)
    
    def select_action(self, state):
        return None
"""
#Value Iteration algorithm
default_values = np.zeros(101)
default_values[0] = 0
default_values[100] = 0
def value_iteration(threshold, V = default_values):

    delta = 0
    compteur = 0
    watcher = True
    ph = 2/5
    for _ in range(100):#while watcher: #threshold < delta:
        for state in range(1, len(V)-1):
            old_v = V[state]
            #V[i] = max([sum(a) for a in action])
            temp_best_va = []
            for action in range(0, min(state, 100 - state)+1):
                if state + action == 100 and state - action == 0:
                    temp_best_va.append((1-ph)*(0 + V[0]) + ph*(1 + V[100]))

                elif state + action == 100:
                    temp_best_va.append((1-ph)*(0 + V[state - action]) + ph*(1 + V[100]))

                elif state - action == 0:
                    temp_best_va.append(ph*(0 + V[state + action]) + (1-ph)*(0 + V[0]))

                else:
                    temp_best_va.append((1-ph)*(0 + V[state - action]) + ph*(0 + V[state + action]))
            V[state] = max(temp_best_va)
            delta = max(delta, abs(old_v - V[state]))
            watcher = threshold < delta
        compteur += 1
        #print(compteur)
        #print(V[99])
    #ouptut deterministic policy
    optimal_policy_per_state = np.zeros(101)
    for state in range(1, len(optimal_policy_per_state)-1):
        temp_best_value_action = 0
        best_policy = 0
        for action in range(1, min(state, 100 - state)+1):
            if state + action == 100 and state - action == 0:
                new_challenger = (1-ph)*(0 + V[0]) + ph*(1 + V[100])

            elif state + action == 100:
                new_challenger = (1-ph)*(0 + V[state - action]) + ph*(1 + V[100])
                    
            elif state - action == 0:
                new_challenger = ph*(0 + V[state + action]) + (1-ph)*(0 + V[0])
            
            else:
                new_challenger = (1-ph)*(0 + V[state - action]) + ph*(0 + V[state + action])
            
            if new_challenger > temp_best_value_action:
                temp_best_value_action = new_challenger
                best_policy = action
        optimal_policy_per_state[state] = best_policy
    #print(optimal_policy_per_state)
    return optimal_policy_per_state, V

if __name__ == "__main__":
    optimal_policy, V_state = value_iteration(10**(-3))
    plt.figure()
    plt.plot(np.arange(1,100), optimal_policy[1:100])
    #plt.plot(np.arange(1,100), V_state[1:100])
    plt.xlabel("States")
    plt.ylabel("Optimal policy")
    plt.show()