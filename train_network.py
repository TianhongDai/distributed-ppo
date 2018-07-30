import arguments
import torch 
import torch.multiprocessing as mp
import dppo_agent
import utils
import gym
import mujoco_py
import models 
from chief import chief_worker  
import os 
os.environ['OMP_NUM_THREADS'] = '1'

# start the main function...
if __name__ == '__main__':
    # get the arguments...
    args = arguments.achieve_args()

    # build up the environment and extract some informations...
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    # define the global network...
    critic_shared_model = models.Critic_Network(num_inputs)
    critic_shared_model.share_memory()

    actor_shared_model = models.Actor_Network(num_inputs, num_actions)
    actor_shared_model.share_memory()

    # define the traffic signal...
    traffic_signal = utils.TrafficLight()
    # define the counter
    critic_counter = utils.Counter()
    actor_counter = utils.Counter()
    # define the shared gradient buffer...
    critic_shared_grad_buffer = utils.Shared_grad_buffers(critic_shared_model)
    actor_shared_grad_buffer = utils.Shared_grad_buffers(actor_shared_model)
    # define shared observation state...
    shared_obs_state = utils.Running_mean_filter(num_inputs)
    # define shared reward...
    shared_reward = utils.RewardCounter()
    # define the optimizer...
    critic_optimizer = torch.optim.Adam(critic_shared_model.parameters(), lr=args.value_lr)
    actor_optimizer = torch.optim.Adam(actor_shared_model.parameters(), lr=args.policy_lr)

    # find how many processor is available...
    num_of_workers = mp.cpu_count() - 1
    processor = []
    workers = []
    
    p = mp.Process(target=chief_worker, args=(num_of_workers, traffic_signal, critic_counter, actor_counter, 
        critic_shared_model, actor_shared_model, critic_shared_grad_buffer, actor_shared_grad_buffer, 
        critic_optimizer, actor_optimizer, shared_reward, shared_obs_state, args.policy_update_step, args.env_name))
    
    processor.append(p)

    
    for idx in range(num_of_workers):
        workers.append(dppo_agent.dppo_workers(args))

    for worker in workers:
        p = mp.Process(target=worker.train_network, args=(traffic_signal, critic_counter, actor_counter, 
            critic_shared_model, actor_shared_model, shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, shared_reward))
        processor.append(p)

    for p in processor:
        p.start()

    for p in processor:
        p.join()


