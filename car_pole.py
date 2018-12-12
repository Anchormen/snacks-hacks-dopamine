import gym
import tensorflow as tf
import numpy as np
from dopamine.agents.dqn.dqn_agent import DQNAgent


slim = tf.contrib.slim
class CarPoleAgent(DQNAgent):

    def __init__(self, session):
        super().__init__(sess=session, num_actions=2, observation_shape=(4,1,1))

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          state: `tf.Tensor`, contains the agent's current state.

        Returns:
          net: _network_type object containing the tensors output by the network.
        """
        net = tf.cast(state, tf.float32)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 2)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=tf.log_sigmoid)
        return self._get_network_type()(q_values)


def expand_observation(observation):
    multi_dim_observation = np.expand_dims(observation, 1)
    multi_dim_observation = np.expand_dims(multi_dim_observation, 2)
    multi_dim_observation = np.expand_dims(multi_dim_observation, 3)
    return multi_dim_observation


with tf.Session() as session:

    env = gym.make('CartPole-v0')
    num_actions = 2 # left or right
    agent = CarPoleAgent(session)

    for i_episode in range(20):
        observation = env.reset()
        observation = expand_observation(observation)
        action = agent.begin_episode(observation)
        for t in range(1000):
            env.render()
            observation, reward, done, info = env.step(action)
            observation = expand_observation(observation)
            action = agent.step(reward, observation)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
