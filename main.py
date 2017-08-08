import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from collections import deque
import numpy as np
from DQN import DQN
import pickle
import time

EPISODE = 1000000
TEST = 100
MAX_STEP_PER_EPISODE = 10000


def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    agent = DQN(env)
    agent.update_target()
    episodes_rewards = [0] * 100
    avg_rewards = []
    skip_rewards = []
    step_num = 0
    for episode in range(EPISODE):
        goal = 0
        img_buf = deque()
        state = env.reset()
        while True:
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            # time.sleep(0.01)
            agent.perceive(state, action, reward, next_state, done, step_num)
            goal += reward
            step_num += 1
            state = next_state
            if done:
                episodes_rewards.pop(0)
                episodes_rewards.append(goal)
                break
                # print "Current reward:", goal," Step number:", step_num
        print("Episode: ", episode, " Last 100 episode average reward: ", np.average(episodes_rewards), " Toal step number: ", step_num, " eps: ", agent.epsilon)

        if step_num > 2000000:
            break

        if episode % 50 == 0:
            skip_rewards.append(goal)

        if episode % 100 == 0:
            avg_rewards.append(np.average(episodes_rewards))
            out_file = open("avg_rewards.pkl",'wb')
            out_file1 = open("skip_rewards.pkl",'wb')
            pickle.dump(avg_rewards, out_file)
            pickle.dump(skip_rewards, out_file1)
            out_file.close()
            out_file1.close()
            agent.saver.save(agent.session, 'saved_networks/' + 'network' + '-dqn', global_step=episode)

    env.close()

def play():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    agent = DQN(env)
    for episode in range(TEST):
        goal = 0
        step_num = 0
        state = env.reset()
        while True:
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            step_num += 1
            env.render()
            time.sleep(0.01)
            goal += reward
            state = next_state
            if done or step_num > MAX_STEP_PER_EPISODE:
                print("Episode: ", episode, " Total reward: ", goal)
                break

if __name__ == '__main__':
    # main()
    play()
