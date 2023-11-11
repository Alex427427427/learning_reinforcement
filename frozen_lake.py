## made by Alexander Li
## 2023-11-11
## for the purpose of learning reinforcement learning

import numpy as np # linear algebra
import gymnasium as gym # game environment
import random # random number generator
import time # for time.sleep()
from IPython.display import clear_output # for clear output

env = gym.make("FrozenLake-v1") # create the environment

action_space_size = env.action_space.n # get the number of actions
state_space_size = env.observation_space.n # get the number of states

q_table = np.zeros((state_space_size, action_space_size)) # create the q-table
print(q_table) # print the q-table

# learning parameters
num_episodes = 10000 # set the number of episodes
max_steps_per_episode = 100 # set the max number of steps per episode

learning_rate = 0.1 # set the learning rate
discount_rate = 0.99 # set the discount rate

exploration_rate = 1 # set the exploration rate
max_exploration_rate = 1 # set the max exploration rate
min_exploration_rate = 0.01 # set the min exploration rate
exploration_decay_rate = 0.001 # set the exploration decay rate

rewards_all_episodes = [] # create a list to store all rewards

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()[0] # reset the environment
    #print(state)
    done = False # set done to false
    rewards_current_episode = 0 # set the current episode reward to 0

    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1) # use a random number to determine whether to explore or exploit
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) # exploit
        else:
            action = env.action_space.sample() # explore

        new_state, reward, done, truncated, info = env.step(action)

        # update the q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    # exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    
    rewards_all_episodes.append(rewards_current_episode)

# calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
for r in rewards_per_thousand_episodes:
    print(f"kiloepoch average reward: {sum(r/1000):.3f}")

# print the updated q-table
print("\n\nQ-table")
print(q_table)

env.close()

env = gym.make("FrozenLake-v1", render_mode="human") # create the environment
# visualise the game in action
#env = gym.make("FrozenLake-v1", render_mode="ansi")

for episode in range(3):
    state = env.reset()[0]
    done = False
    print(f"Episode {episode + 1}\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("You reached the goal!")
                time.sleep(3)
            else:
                print("You fell through a hole!")
                time.sleep(3)
            clear_output(wait=True)
            break

        state = new_state

env.close()