import torch
import time
from torch.autograd import Variable

def chief_worker(num_workers, traffic_signal, critic_counter, actor_counter, critic_shared_model, actor_shared_model, \
    critic_shared_grad_buffer, actor_shared_grad_buffer, critic_optimizer, actor_optimizer, shared_reward, shared_obs_state, update_step, name):
    num_iteration = 1
    while True:
        time.sleep(1)
        # the chief will update the whole network when it receive the enough gradients..
        if critic_counter.get() >= num_workers:
            for n, p in critic_shared_model.named_parameters():
                p.grad = Variable(critic_shared_grad_buffer.grads[n + '_grad'])
            
            # start to update the critic network
            critic_optimizer.step()
            # clean the buffer....
            critic_counter.reset()
            critic_shared_grad_buffer.reset()
            traffic_signal.switch()
        
        # start to update the actor network...
        if actor_counter.get() >= num_workers:
            for n, p in actor_shared_model.named_parameters():
                p.grad = Variable(actor_shared_grad_buffer.grads[n + '_grad'])
            
            # start to reset the buffer....
            actor_optimizer.step()

            # get the reward...
            if num_iteration % update_step == 0:
                reward_batch = shared_reward.get()
                reward_batch /= num_workers
                shared_reward.reset()
                print('The iteration is ' + str(int(num_iteration/update_step)) + ' and the reward mean is ' + str(reward_batch))

            if num_iteration % (update_step * 10) == 0:
                save_path = 'saved_models/' + name + '/models_' + str(int(num_iteration/update_step)) + '.pt'
                torch.save([actor_shared_model.state_dict(), shared_obs_state.get_results()], save_path)

            num_iteration += 1
            actor_counter.reset()    
            actor_shared_grad_buffer.reset()
            traffic_signal.switch()

            
