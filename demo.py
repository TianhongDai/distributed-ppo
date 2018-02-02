import arguments
import dppo_agent

# achieve the arguments..
args = arguments.achieve_args()

# build th worker...
worker = dppo_agent.dppo_workers(args)
model_path = 'saved_models/' + args.env_name + '/model.pt'
worker.test_network(model_path)
