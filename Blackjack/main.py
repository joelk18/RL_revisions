from Classes import Agent, Environment
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import numpy as np

#Initialize agent & environment
agent = Agent()
env = Environment(17)

def episode():
    prev_state = env.start()
    terminate = prev_state[3]
    prev_action = agent.action(prev_state)
    #breakpoint()
    while not terminate:
        state = env.play(prev_action)
        action = agent.action(state)
        assert prev_state[0] < 21
        agent.update_value_action_table(state, action, prev_state, prev_action)
        terminate = state[3]
        #breakpoint()
        prev_state = state
        prev_action = action
    #reinitialize the env values
    env.reset_state()

def plot_optimal_policy(matrix):
    """
    Plots the optimal policy for each row in the matrix.
    The action is "hit" if matrix[row, col, 1] > matrix[row, col, 0].
    """
    rows, cols, _ = matrix.shape
    for row in range(rows):
        policy = []
        for col in range(cols):
            # Determine the action: "hit" if value at index 1 > value at index 0
            if matrix[row, col, 1] > matrix[row, col, 0]:
                policy.append(1)  # "hit"
            else:
                policy.append(0)  # "stick"
        
        # Plot the policy for the current row
        plt.figure()
        plt.plot(list(range(4, 22)), policy, marker='o', linestyle='-', label=f'{row+2}')
        plt.title(f'Optimal Policy for value {row+2}')
        plt.xlabel('Column')
        plt.ylabel('Action (0=Stick, 1=Hit)')
        plt.ylim(-0.5, 1.5)
        plt.yticks([0, 1], labels=["Stick", "Hit"])
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    for i in range(1000000):
        episode()
        if i%10000==0:
            print("-------------------")
            print(f"Episode {i} done")
    
    # Plot the optimal policy
    print(agent.value_action[:, -1, :])
    #plot_optimal_policy(agent.value_action)
    #save the value action table in a csv file
    for i in range(10):
        np.savetxt(f"value_action_table/value_action_table_{i}.csv", agent.value_action[i, :, :], delimiter=",")
