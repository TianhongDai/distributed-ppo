# Distributed Proximal Policy Optimization (DPPO)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This is an pytorch-version implementation of [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286). This project is based on [Alexis David Jacq's DPPO project](https://github.com/alexis-jacq/Pytorch-DPPO). However, it has been rewritten and contains some modifications that appaer to improve learning in some environments. In this code, I revised the Running Mean Filter and this leads to better performance (for example in Walker2D). I also rewrote the code to support the **Actor Network** and **Critic Network** separately. This change then  allows the creation of [asymmetric](https://arxiv.org/abs/1710.06542) for some tasks, where the information available at training time is not available at run time. Further, the actions in this project are sampled from a Beta Distribution, leads to better training speed and performance in a large number of tasks.

## Requirements

- python 3.5.2
- openai gym
- mujoco-python
- pytorc-cpu(***Please use the CPU(None-CUDA) version!!! --- I will solve the problem in the GPU(CUDA) version later***)
- [pyro](http://pyro.ai/)

## Instruction to run the code
### Train your models
    cd /root-of-this-code/
    python train_network.py

You could also try other mujoco's environments. This code has already pre-trained one mujoco environment: `Walker2d-v1`. You could try it by yourself on your favourite task!

### Test your models:
    cd /root-of-this-code/
    python demo.py

## Results
### Training Curve
![img](https://github.com/TianhongDai/Distributed_PPO/blob/master/results/training_plot.png)
### Demo: Walker2d-v1
![img](https://github.com/TianhongDai/Distributed_PPO/blob/master/results/walker2d.gif)

## Acknowledgement
- [Alexis David Jacq's DPPO](https://github.com/alexis-jacq/Pytorch-DPPO)

## Reference
[1] [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) 





