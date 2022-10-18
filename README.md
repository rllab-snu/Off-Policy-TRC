# Efficient Off-Policy Safe Reinforcement Learning Using Trust Region Conditional Value at Risk

This is an official GitHub Repository for paper ([link](https://doi.org/10.1109/LRA.2022.3184793)):

- Dohyeong Kim and Songhwai Oh, "Efficient Off-Policy Safe Reinforcement Learning Using Trust Region Conditional Value at Risk," IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7644-7651, Jul. 2022.

- This paper proposes an off-policy version of CVaR-constrained safe RL method (called **Off-Policy TRC**).

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Requirement

- python 3.7 or greater
- gym
- mujoco-py (https://github.com/openai/mujoco-py)
- safety-gym (https://github.com/openai/safety-gym)
- stable-baselines3
- tensorflow-gpu==2.4.0
- tensorflow-probability==0.12.2
- torch==1.9.0
- requests
- wandb

## Supported environment list

- `HalfCheetah-v2` (*half_cheetah*)
- `Walker2-v2` (*walker*)
- `Safexp-PointGoal1-v0` (*point_goal*)
- `Safexp-CarGoal1-v0` (*car_goal*)
- `Safexp-DoggoGoal4-v0` (*doggo_goal*, defined in `utils/register.py`)
- `Jackal-v2` (*jackal*, defined in `utils/jackal_env/env3.py`)

## How to use

- The results in the paper was recorded by `tf1_ver0` but improved in `tf1_ver1`. 
- The `torch` version is not optimized currently, so performance can be degraded. 

### tf1

- training:

  - ```bash
    cd tf1_ver1
    bash train/{env_name}.sh
    ```

- test:

  - ```bash
    cd tf1_ver1
    bash test/{env_name}.sh
    ```

### torch

- training:

  - ```bash
    cd torch
    bash train/{env_name}.sh
    ```

- test:

  - ```bash
    cd torch
    bash test/{env_name}.sh
    ```
