# Distributed Proximal Policy Optimization (DPPO)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This is an pytorch-version implementation of [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286). This project is based on [Alexis David Jacq's DPPO project](https://github.com/alexis-jacq/Pytorch-DPPO). In this code, I revised the Running Mean Filter and it could have better performance than the origion code. I also make the **Actor Network** and **Critic Network** separately which could be expanded as [Asymmetric Actor Critic Structure]() for some special tasks. The actions in this project are sampled from Beta Distribution.

## Requirements

- python 3.5.2
- openai gym
- mujoco-python
- pytorch
- [pyro](http://pyro.ai/)

## Instruction to run the code
### Train your models
```bash
cd /root-of-this-code/
python main.py

```
You could also try some other mujoco's environment. This code has already pre-trained one mujoco environments: `Walker2d-v1`. You could try them by yourself!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py

```

## Acknowledgement
- [Alexis David Jacq's DPPO](https://github.com/alexis-jacq/Pytorch-DPPO)

## Reference
[1] [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) 





