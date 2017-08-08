import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from collections import deque
import baselines.common.tf_util as U


GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32
UPDATE_SEQ = 1000
TRAIN_SEQ = 4

class DQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_dim = env.action_space.n

        self.state_input, self.q_value = self.create_network("q_func")
        self.tar_state_input, self.tar_q_value = self.create_network("tar_q_func")
        self.update_target = self.create_update_target()

        self.create_training_method()

        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def create_network(self, scope):
        with tf.variable_scope(scope, reuse=False):
            state_input = tf.placeholder('float', [None, 84, 84, 4])
            out = layers.convolution2d(state_input, num_outputs=32, kernel_size=8, stride=1, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

            conv_out = layers.flatten(out)
            value_out = layers.fully_connected(conv_out, num_outputs=256, activation_fn=tf.nn.relu)
            q_value = layers.fully_connected(value_out, num_outputs=self.action_dim, activation_fn=None)
            return state_input, q_value

    def create_update_target(self):
        q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="tar_q_func")
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)
        update_target = U.function([], [], updates=[update_target_expr])
        return update_target

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        self.y_input = tf.placeholder("float", [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done, step_num):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) >= REPLAY_SIZE and step_num % TRAIN_SEQ == 0:
            self.train_network()

        if len(self.replay_buffer) >= REPLAY_SIZE and step_num % UPDATE_SEQ == 0:
            self.update_target()

    def train_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        for state, action, reward, next_state, _ in minibatch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)

        y_batch = []
        q_value_batch = self.tar_q_value.eval(feed_dict={self.tar_state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        q_value = self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 200000
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)

        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
