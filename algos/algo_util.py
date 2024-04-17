import tensorflow as tf

# import tensorflow.keras.layers as layers
import numpy as np
from gym import spaces

from maddpg.trainer.maddpg import MADDPGAgentTrainer as MADDPGAgentActor

import tensorflow.compat.v1 as tf_v1
tf_v1.disable_eager_execution()
import tf_slim

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf_v1.variable_scope(scope, reuse=reuse):
        out = input
        out = tf_slim.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = tf_slim.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = tf_slim.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_actors(env, n_agents, arglist):
    actors = []
    model = mlp_model
    actor = MADDPGAgentActor
    obs_shape_n = [(env.get_obs_size(),) for _ in range(n_agents)]
    for i in range(n_agents):
        actors.append(actor(
            "agent_%d" % i, model, obs_shape_n, env.action_space(), i, arglist))
    return actors

def get_leader_actors(env, n_groups, arglist):
    actors = []
    model = mlp_model
    actor = MADDPGAgentActor
    obs_shape_n = [(env.get_obs_size() * 2, ), (env.get_obs_size() * 7, ), (env.get_obs_size(), )]
    for i in range(n_groups):
        actors.append(actor(
            "leader_%d" % i, model, obs_shape_n, [spaces.Discrete(3) for _ in range(n_groups)], i, arglist))
    return actors

def get_actions(actors, obs, env, side):
    action_n = [agent.action(obs) for agent, obs in zip(actors, obs)]
    action_for_smac = [np.argmax(action_ar) for action_ar in action_n]
    action_for_smac = [action if env.get_avail_agent_actions(agent, side=side)[action] else
                       np.nonzero(env.get_avail_agent_actions(agent, side=side))[0][-1] for agent, action in
                       enumerate(action_for_smac)]
    # TODO(alan): Redundant?
    action_for_smac = [action if env.is_agent_alive(agent, side=side) else 0 for agent, action in
                       enumerate(action_for_smac)]
    return action_for_smac