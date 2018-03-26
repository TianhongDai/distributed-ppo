import numpy as np
import torch
from torch.autograd import Variable
import models
import pyro
import pyro.distributions as dist
import gym
import mujoco_py

# start to define the workers...
class dppo_workers:
    def __init__(self, args):
        self.args = args 
        self.env = gym.make(self.args.env_name)

        # get the numbers of observation and actions...
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        # define the network...
        self.actor_net = models.Actor_Network(num_inputs, num_actions)
        self.critic_net = models.Critic_Network(num_inputs)
    
    # start to define the training function...
    def train_network(self, traffic_signal, critic_counter, actor_counter, shared_critic_model, shared_actor_model, \
                                shared_obs_state, critic_shared_grad_buffer, actor_shared_grad_buffer, reward_buffer):

        # update the parameters....
        self.actor_net.load_state_dict(shared_actor_model.state_dict())
        self.critic_net.load_state_dict(shared_critic_model.state_dict())
        while True:
            # update the parameters...
            # define the memory...
            brain_memory = []
            reward_sum = 0
            for _ in range(self.args.collection_length):
                state = self.env.reset()
                state = shared_obs_state.normalize(state)
                while True:
                    # put the state into the Variables....
                    state_tensor = Variable(torch.Tensor(state).unsqueeze(0))
                    # input the state into the network to predict the actions...
                    action_alpha, action_beta = self.actor_net(state_tensor)
                    # sample actions from the beta distribution....
                    actions_cpu, actions_real = self.select_actions(action_alpha, action_beta)
                    # input actions into the environment...
                    state_, reward, done, _ = self.env.step(actions_real)
                    # accumulate the rewards...
                    reward_sum += reward
                    # start to store the trainsition...
                    brain_memory.append((state, reward, done, actions_cpu))
                    if done:
                        break 
                
                    # normalize the state...
                    state_ = shared_obs_state.normalize(state_)
                    state = state_
            # start to calculate the gradients for this time sequence...
            reward_buffer.add(reward_sum / self.args.collection_length)
            critic_loss, actor_loss = self.update_network(brain_memory, critic_shared_grad_buffer, actor_shared_grad_buffer, \
                                            shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal)


    # calculate the gradients based on the information be collected...
    def update_network(self, brain_memory, critic_shared_grad_buffer, actor_shared_grad_buffer, \
                             shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):
        # process the stored information
        state_batch = torch.Tensor(np.array([element[0] for element in brain_memory]))
        reward_batch = torch.Tensor(np.array([element[1] for element in brain_memory]))
        done_batch = [element[2] for element in brain_memory]
        actions_batch = torch.Tensor(np.array([element[3] for element in brain_memory]))

        # put them into the Variables...
        state_batch_tensor = Variable(state_batch)
        actions_batch_tensor = Variable(actions_batch)
        # calculate the discounted reward...
        returns, advantages, old_action_prob = self.calculate_discounted_reward(state_batch_tensor, \
                                                            done_batch, reward_batch, actions_batch_tensor)

        # calculate the gradients...
        critic_loss, actor_loss = self.calculate_the_gradients(state_batch_tensor, actions_batch_tensor, \
                                returns, advantages, old_action_prob, critic_shared_grad_buffer, actor_shared_grad_buffer, \
                                shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal)
        
        return critic_loss.data.cpu().numpy()[0], actor_loss.data.cpu().numpy()[0]

    # calculate the gradients...
    def calculate_the_gradients(self, state_batch_tensor, actions_batch, returns, advantages, old_action_prob, critic_shared_grad_buffer, \
                            actor_shared_grad_buffer, shared_critic_model, shared_actor_model, critic_counter, actor_counter, traffic_signal):

        # put the tensors into the Variable...
        returns = Variable(returns)
        advantages = Variable(advantages)
        # start to calculate the gradient of critic network firstly....
        for _ in range(self.args.value_update_step):
            self.critic_net.zero_grad()
            # get the init signal...
            signal_init = traffic_signal.get()
            # start to process...
            predicted_value = self.critic_net(state_batch_tensor)
            # calculate the critic loss firstly...
            critic_loss = (returns - predicted_value).pow(2).mean()
            # do the back-propagation...
            critic_loss.backward()
            # add the gradient to the shared_buffer...
            critic_shared_grad_buffer.add_gradient(self.critic_net)
            # after add the gradient, add the counter...
            critic_counter.increment()
            # wait for the cheif's signal...
            while signal_init == traffic_signal.get():
                pass
            self.critic_net.load_state_dict(shared_critic_model.state_dict())
        
        # start to update the critic_network....
        for _ in range(self.args.policy_update_step):
            # get the init signal....
            self.actor_net.zero_grad()
            signal_init = traffic_signal.get()
            # start to process...
            action_alpha, action_beta = self.actor_net(state_batch_tensor)
            new_beta_dist = dist.Beta(action_alpha, action_beta)
            new_action_prob = new_beta_dist.batch_log_pdf(actions_batch)
            ratio = torch.exp(new_action_prob - old_action_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # do the back propogation
            actor_loss.backward()
            actor_shared_grad_buffer.add_gradient(self.actor_net)
            actor_counter.increment()
            while signal_init == traffic_signal.get():
                pass
            self.actor_net.load_state_dict(shared_actor_model.state_dict())
        
        return critic_loss, actor_loss

    # calculate the discounted reward
    def calculate_discounted_reward(self, state_batch_tensor, done_batch, reward_batch, actions_batch_tensor):
        # calculate the predicted value firstly...
        predicted_value = self.critic_net(state_batch_tensor)
        # calculate the returns and advantages firstly...
        predicted_value = predicted_value.detach()

        returns = torch.Tensor(len(done_batch), 1)
        advantages = torch.Tensor(len(done_batch), 1)
        deltas = torch.Tensor(len(done_batch), 1)

        previous_returns = 0
        previous_advantages = 0
        previous_value = 0
        # use gae here...
        for idx in reversed(range(len(done_batch))):
            if done_batch[idx]:
                returns[idx, 0] = reward_batch[idx]
                #deltas[idx, 0] = reward_batch[idx] - predicted_value.data[idx, 0]
                #advantages[idx, 0] = deltas[idx, 0]
                advantages[idx, 0] = returns[idx, 0] - predicted_value.data[idx, 0]
            else:
                returns[idx, 0] = reward_batch[idx] + self.args.gamma * previous_returns
                #deltas[idx, 0] = reward_batch[idx] + self.args.gamma * previous_value - predicted_value.data[idx, 0]
                #advantages[idx, 0] = deltas[idx, 0] + self.args.gamma * self.args.tau * previous_advantages
                advantages[idx, 0] = returns[idx, 0] - predicted_value.data[idx, 0]

            previous_returns = returns[idx, 0]
            previous_value = predicted_value.data[idx, 0]
            previous_advantages = advantages[idx, 0]

        # normalize the advantages...
        advantages = (advantages - advantages.mean()) / advantages.std()
        # calculate the old action probabilities...
        action_alpha, action_beta = self.actor_net(state_batch_tensor)
        old_beta_dist = dist.Beta(action_alpha, action_beta)
        old_action_prob = old_beta_dist.batch_log_pdf(actions_batch_tensor)
        old_action_prob = old_action_prob.detach()
        
        return returns, advantages, old_action_prob 

    # sample actions from the beta distributions....
    def select_actions(self, alpha, beta):
        actions = dist.beta(alpha, beta)
        actions_cpu = actions.data.cpu().numpy()[0]
        # real action...
        actions_real = actions_cpu.copy()
        actions_real = -1 + actions_real * 2

        return actions_cpu, actions_real

# ------------------------------------------------------------------------------------------#
# HERE, WE STRAT TO TEST OUR ALGORITHMS...
    def test_network(self, model_path):
        # load the models and means and std...
        policy_model, running_mean_filter = torch.load(model_path, map_location=lambda storage, loc: storage)
        mean = running_mean_filter[0]
        std = running_mean_filter[1]

        self.actor_net.load_state_dict(policy_model)
        self.actor_net.eval()

        # start to test...
        while True:
            state = self.env.reset()
            state = self.normalize_filter(state, mean, std)
            reward_sum = 0
            while True:
                self.env.render()
                state_tensor = Variable(torch.Tensor(state).unsqueeze(0))
                # input the state into the network...
                action_alpha, action_beta = self.actor_net(state_tensor)
                # build up the beta distribution...
                action = dist.Beta(action_alpha, action_beta).analytic_mean()
                action_real = action.data.cpu().numpy()[0]
                action_real = -1 + 2 * action_real
                # input the action into the environment...
                state_, reward, done, _ = self.env.step(action_real)
                # sum the reward...
                reward_sum += reward
                if done:
                    break 
                state_ = self.normalize_filter(state_, mean, std)
                state = state_

            print('the reward sum in this episode is ' + str(reward_sum) + '!')


    # this is used in the testing...
    def normalize_filter(self, x, mean, std):
        x = (x - mean) / (std + 1e-8)
        x = np.clip(x, -5.0, 5.0)

        return x

