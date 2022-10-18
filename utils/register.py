from safety_gym.envs.suite import SafexpEnvBase
from gym.envs.registration import register

bench_base = SafexpEnvBase('', {'observe_goal_lidar': False,
                                'observe_goal_dist': True,
                                'observe_goal_comp': True,
                                'observe_box_lidar': True,
                                'lidar_max_dist': 3,
                                'lidar_num_bins': 16
                                })

goal_all = {
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    }

goal4 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'hazards_num': 8,
    'vases_num': 0,
    'constrain_hazards': True,
    'observe_hazards': True,
}

bench_goal_base = bench_base.copy('Goal', goal_all)
bench_goal_base.register('4', goal4)

# for jackal
register(
    id='Jackal-v0',
    entry_point='utils.jackal_env.env:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Jackal-v1',
    entry_point='utils.jackal_env.env2:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)

register(
    id='Jackal-v2',
    entry_point='utils.jackal_env.env3:Env',
    max_episode_steps=1000,
    reward_threshold=10000.0,
)
