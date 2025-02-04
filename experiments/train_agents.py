import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from smac.env import StarCraft2Env
import matplotlib.pyplot as plt


def draw_won_rate(won_rate):
    plt.xlabel('episodes')
    plt.ylabel('win rate')
    plt.plot([100 * i for i in range(1, len(won_rate) + 1)], won_rate)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # TODO(alan): change to pass from smac_maps's config
    parser.add_argument("--max-episode-len", type=int, default=150, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
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
    parser.add_argument("--save-dir", type=str, default="/home/alantang/agents_policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(env, n_agents, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(n_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space(), i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    # TODO(alan) : Set multi-cpu to boost training
    with U.multi_threaded_session():
        # Create environment
        env = StarCraft2Env(map_name="MMM_redirect_train", difficulty='5')
        env_info = env.get_env_info()

        n_agents = env_info["n_agents"]
        # env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        obs_shape_n = [(env.get_obs_size(),) for _ in range(n_agents)]
        # num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, n_agents, obs_shape_n, arglist)

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_win = []
        episode_killing = []
        episode_remaining = []
        agent_rewards = [[0.0] for _ in range(n_agents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        won_rate = []
        buffer_len = 0

        print('Starting iterations...')
        while True:
            obs_n, _ = env.get_obs()
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            action_for_smac = [np.argmax(action_ar) for action_ar in action_n]
            action_for_smac = [action if env.get_avail_agent_actions(agent)[action] else np.nonzero(env.get_avail_agent_actions(agent))[0][-1] for agent, action in enumerate(action_for_smac)]
            action_for_smac = [action if env.is_agent_alive(agent) else 0 for agent, action in enumerate(action_for_smac)]
            # environment step
            # new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            rew_n, terminal, info = env.step([action_for_smac])
            rew_n = list(rew_n)
            # TODO(alan): set individual reward
            # rew_n = [(rew / n_agents) for _ in range(n_agents)]
            new_obs_n, _ = env.get_obs()
            done_n = [False for _ in range(n_agents)]
            episode_step += 1
            done = all(done_n)
            # terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                buffer_len += 1
            # obs_n = new_obs_n

            # print(f'at ep {len(episode_rewards)} rew_n len: {len(rew_n)}')
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                game_restart = not ('dead_enemies' in info.keys())
                enemy_killed_num = info.get('dead_enemies', 0)
                self_left_num = n_agents - info.get('dead_allies', 0)
                if not game_restart:
                    episode_killing.append(enemy_killed_num)
                    episode_remaining.append(self_left_num)
                    episode_win.append(1 if info['battle_won'] else 0)
                env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                latest_won_rate = round(np.mean(episode_win[-arglist.save_rate:]), 2)
                won_rate.append(latest_won_rate)
                U.save_state(arglist.save_dir, latest_won_rate, saver)
                # TODO(alan): check the difference
                # print statement depends on whether or not there are adversaries
                # if num_adversaries == 0:
                #     print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                # else:
                print("steps: {}, episodes: {}, mean won rate: {}, mean episode reward: {}, "
                      "agent episode reward: {}, mean episode killing: {}, mean_episode remaining: {}, time: {}".format(
                        train_step,
                        len(episode_rewards),
                        latest_won_rate,
                        round(np.mean(episode_rewards[-arglist.save_rate:]), 1),
                        [round(np.mean(rew[-arglist.save_rate:]), 1) for rew in agent_rewards],
                        round(np.mean(episode_killing[-arglist.save_rate:]), 2),
                        round(np.mean(episode_remaining[-arglist.save_rate:]), 2),
                        round(time.time()-t_start, 2)))
                print(f'buffer len: {buffer_len}')

                env.save_replay(latest_won_rate)

                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

            if len(episode_rewards) == 50000: break
    env.close()
    draw_won_rate(won_rate)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
