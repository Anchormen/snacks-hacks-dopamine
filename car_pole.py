import gym
import tensorflow as tf
from dopamine.agents.dqn.dqn_agent import DQNAgent


with tf.Session() as session:

    env = gym.make('CartPole-v0')
    num_actions = 2 # left or right
    agent = DQNAgent(
        sess=session,
        num_actions=num_actions, # THE CONVNET COMPLAINS ABOUT THIS!
        observation_shape=(4,)
    )

    for i_episode in range(20):
        observation = env.reset()
        action = agent.begin_episode(observation)
        for t in range(100):
            env.render()
            observation, reward, done, info = env.step(action)

            action = agent.step(reward, observation)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
