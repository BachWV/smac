import argparse

from smac.env import StarCraft2Env

import maddpg.common.tf_util as U
import tensorflow as tf
import numpy as np
from algos.algo_util import get_actors, get_actions, get_leader_actors

from experiments.train_group_leader import p2a
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # TODO(alan): change to pass from smac_maps's config
    parser.add_argument("--max-episode-len", type=int, default=150, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='maddpg_in_smac', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/root/smac/experiments/policy/agents_policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="C:\\Users\\Charon\\Desktop\\code\\star\\smacTang\\experiments\\policy\\agents_policy\\benchmark_files\\", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="C:\\Users\\Charon\\Desktop\\code\\star\\smacTang\\experiments\\policy\\agents_policy\\learning_curves\\", help="directory where plot data is saved")
    return parser.parse_args()
def main():
    #env = StarCraft2Env(map_name="MMM_redirect_train", policy_agents_num=2)
    env = StarCraft2Env(map_name="MMM_redirect_train", difficulty='5')
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    n_groups = 3

    n_episodes = 100
    win_cnt = 0

    g1 = tf.Graph()
    g2 = tf.Graph()

    #leaders_sess = tf.Session(graph=g1)
    leaders_sess = tf.compat.v1.Session(graph=g1)
    agents_sess = tf.compat.v1.Session(graph=g2)
    arglist = parse_args()
    with leaders_sess.as_default():
        with g1.as_default():
            leaders = get_leader_actors(env, n_groups, arglist)
            U.initialize()
            U.load_state("/root/smac/experiments/policy/leaders_policy/")
            print("load leaders")


    with agents_sess.as_default():
        with g2.as_default():

            red_actors = get_actors(env, n_agents, arglist)
            blue_actors = red_actors[:]

            U.initialize()
            U.load_state("/root/smac/experiments/policy/agents_policy/")
            print("load agents state")

    for e in range(n_episodes):
        print("before reset")
        env.reset()
        print("after reset")
        terminal = False
        ep_step = 0

        while not terminal:
            with agents_sess.as_default():
                with g2.as_default():
                    red_obs, blue_obs = env.get_obs()

                    red_act = get_actions(red_actors, red_obs, env, 'red')
                    blue_act = get_actions(blue_actors, blue_obs, env, 'blue')

            if ep_step % 5 == 0:
                with leaders_sess.as_default():
                    with g1.as_default():
                        obs_n = env.get_obs_leader_n(side='red')
                        policy_vec_n = [leader.action(obs) for leader, obs in zip(leaders, obs_n)]
                        policy_int_n = [np.argmax(policy_vec) for policy_vec in policy_vec_n]
                        # get action
                        red_act = p2a(policy_int_n, env, side='red')

            # environment step
            _, terminal, info = env.step([red_act, blue_act])
            ep_step += 1

        # TODO(alan): Now if time runs up, red_side lose regardless of how many agent left?
        if info['battle_won']:
            print('win game {}'.format(e))
            win_cnt += 1
        else:
            print('lose game {}'.format(e))
        env.close()
        print('current win rate {}'.format(win_cnt / (e + 1)))

    print('win rate {}'.format(win_cnt / n_episodes))


if __name__ == "__main__":
    main()
