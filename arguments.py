import argparse
import torch

# define some arguments that will be used...
def achieve_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--seed', type=int, default=123, help='the random seed')
    parse.add_argument('--policy_lr', type=float, default=3e-4, help='the learning rate of actor network')
    parse.add_argument('--value_lr', type=float, default=3e-4, help='the learning rate of critic network')
    parse.add_argument('--batch_size', type=int, default=1, help='the batch size of the training')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount ratio...')
    parse.add_argument('--policy_update_step', type=int, default=10, help='the update number of actor network')
    parse.add_argument('--value_update_step', type=int, default=10, help='the update number of critic network')
    parse.add_argument('--epsilon', type=float, default=0.2, help='the clipped ratio...')
    parse.add_argument('--tau', type=float, default=0.95, help='the coefficient for calculate GAE')
    parse.add_argument('--max_episode_length', type=int, default=100, metavar='LENGTH', help='Maximum episode length')
    parse.add_argument('--env_name', default='Walker2d-v1', help='environments name')    
    parse.add_argument('--collection_length', type=int, default=8, help='the sample collection length(episodes)')

    args = parse.parse_args()

    return args


