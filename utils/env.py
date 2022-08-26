from scipy.spatial.transform import Rotation
import numpy as np
import safety_gym
import pickle
import random
import gym
import cv2
import re

class GymEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        _, self.robot_name, self.task_name = re.findall('[A-Z][a-z]+', env_name)
        self.robot_name = self.robot_name.lower()
        self.task_name = self.task_name.lower()
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if self.task_name == 'goal':
            self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16
        elif self.task_name == 'button':
            self.obs_dim = 2 + 1 + 2 + 2 + 1 + 16*3
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = self._env.action_space
        self.goal_threshold = np.inf
        self.hazard_size = 0.2
        self.gremlin_size = 0.1*np.sqrt(2.0)
        self.button_size = 0.1
        self.safety_confidence = 0.0


    def seed(self, num_seed):
        num_seed += random.randint(0, 10000)
        self._env.seed(num_seed)

    def _get_original_state(self):
        goal_dir = self._env.obs_compass(self._env.goal_pos)
        goal_dist = np.array([self._env.dist_goal()])
        goal_dist = np.clip(goal_dist, 0.0, self.goal_threshold)
        acc = self._env.world.get_sensor('accelerometer')[:2]
        vel = self._env.world.get_sensor('velocimeter')[:2]
        rot_vel = self._env.world.get_sensor('gyro')[2:]
        if self.task_name == 'goal':
            hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
            lidar = hazards_lidar
        elif self.task_name == 'button':
            hazards_lidar = self._env.obs_lidar(self._env.hazards_pos, 3)
            gremlins_lidar = self._env.obs_lidar(self._env.gremlins_obj_pos, 3)
            buttons_lidar = self._env.obs_lidar(self._env.buttons_pos, 3)
            lidar = np.concatenate([hazards_lidar, gremlins_lidar, buttons_lidar])
        state = np.concatenate([goal_dir/0.7, (goal_dist - 1.5)/0.6, acc/8.0, vel/0.2, rot_vel/2.0, (lidar - 0.3)/0.3], axis=0)
        return state

    def _get_cost(self, h_dist):
        h_coeff = 10.0
        cost = 1.0/(1.0 + np.exp((h_dist - self.safety_confidence)*h_coeff))
        return cost

    def _get_min_dist(self, hazard_pos_list, pos):
        pos = np.array(pos)
        min_dist = np.inf
        for hazard_pos in hazard_pos_list:
            dist = np.linalg.norm(hazard_pos[:2] - pos[:2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_hazard_dist(self):
        if self.task_name == "button":
            # button
            pos_list = []
            for button_pos in self._env.buttons_pos:
                rel_pos = button_pos - self._env.goal_pos
                if np.linalg.norm(rel_pos) < 1e-5:
                    continue
                pos_list.append(button_pos)
            h_dist = self._get_min_dist(pos_list, self._env.world.robot_pos()) - self.button_size

            # gremlin
            temp_dist = self._get_min_dist(self._env.gremlins_obj_pos, self._env.world.robot_pos()) - self.gremlin_size
            h_dist = min(h_dist, temp_dist)

            # hazard
            temp_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
            h_dist = min(h_dist, temp_dist)
        elif self.task_name == "goal":
            # hazard
            h_dist = self._get_min_dist(self._env.hazards_pos, self._env.world.robot_pos()) - self.hazard_size
        return h_dist

    def get_step_wise_cost(self):
        h_dist = self._get_hazard_dist()
        step_wise_cost =  self.safety_confidence - h_dist
        return step_wise_cost
        
    def reset(self):
        self.t = 0
        self._env.reset()
        state = self._get_original_state()
        return state

    def step(self, action):
        reward = 0
        is_goal_met = False
        num_cv = 0

        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)

            if info['cost'] > 0:
                num_cv += 1

            try:
                if info['goal_met']:
                    is_goal_met = True
            except:
                pass
                
            reward += r_t
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break

        state = self._get_original_state()
        h_dist = self._get_hazard_dist()

        info['goal_met'] = is_goal_met
        info['cost'] = self._get_cost(h_dist)
        info['num_cv'] = num_cv
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()


class GymEnv4(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

        self.hazard_size = 0.2
        self.safety_confidence = 0.0
        self.hazard_size_confidence = self.hazard_size + self.safety_confidence

        self.obs_dim = self._env.observation_space.shape[0]
        self.action_dim = self._env.action_space.shape[0]
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = gym.spaces.box.Box(-np.ones(self.action_dim), np.ones(self.action_dim))


    def seed(self, num_seed):
        self._env.seed(num_seed)

    def getCost(self, h_dist):
        h_coeff = 10.0
        cost = 1.0/(1.0 + np.exp((h_dist - self.hazard_size_confidence)*h_coeff))
        return cost

    def getMinDist(self, hazard_pos_list, pos):
        pos = np.array(pos)
        min_dist = np.inf
        for hazard_pos in hazard_pos_list:
            dist = np.linalg.norm(hazard_pos[:2] - pos[:2])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def getHazardsDist(self):
        h_dist = self.getMinDist(self._env.hazards_pos, self._env.world.robot_pos())
        return h_dist
        
    def reset(self):
        self.t = 0
        state = self._env.reset()
        return state

    def step(self, action):
        reward = 0
        is_goal_met = False
        num_cv = 0
        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)
            if info['cost'] > 0:
                num_cv += 1
            try:
                if info['goal_met']:
                    is_goal_met = True
            except:
                pass                
            reward += r_t
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break
        state = s_t
        h_dist = self.getHazardsDist()
        info['goal_met'] = is_goal_met
        info['cost'] = self.getCost(h_dist)
        info['num_cv'] = num_cv
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()

class CheetahEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

        self.obs_dim = self._env.observation_space.shape[0]
        self.action_dim = self._env.action_space.shape[0]
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = gym.spaces.box.Box(-np.ones(self.action_dim), np.ones(self.action_dim))


    def seed(self, num_seed):
        self._env.seed(num_seed)

    def getCost(self, state):
        a = 45.0*np.pi/180.0
        b = 10.0
        rooty = state[1]
        cost = 1.0/(1.0 + np.exp((a - np.abs(rooty))*b))
        return cost

    def reset(self):
        self.t = 0
        state = self._env.reset()
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)
            reward += r_t
            self.t += 1
            done = d_t or self.t == self.max_episode_length
            if done:
                break
        state = s_t
        info['goal_met'] = False
        info['cost'] = self.getCost(state)
        info['num_cv'] = 1 if info['cost'] >=0.5 else 0
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()

class WalkerEnv(gym.Env):
    def __init__(self, env_name, seed, max_episode_length, action_repeat):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat

        self.obs_dim = self._env.observation_space.shape[0]
        self.action_dim = self._env.action_space.shape[0]
        self.observation_space = gym.spaces.box.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim))
        self.action_space = gym.spaces.box.Box(-np.ones(self.action_dim), np.ones(self.action_dim))


    def seed(self, num_seed):
        self._env.seed(num_seed)

    def getCost(self, state):
        a = 20.0*np.pi/180.0
        b = 20.0
        rooty = state[1] # -0.2 < rooty< 0.2
        cost1 = 1.0/(1.0 + np.exp((a - np.abs(rooty))*b))

        a = 0.5
        b = 15.0
        height = state[0] # 0.7 < height
        cost2 = 1.0/(1.0 + np.exp((height - a)*b))

        cost = cost2
        return cost

    def reset(self):
        self.t = 0
        state = self._env.reset()
        return state

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            s_t, r_t, d_t, info = self._env.step(action)
            reward += (r_t - 1.0)
            self.t += 1
            done = self.t == self.max_episode_length
            if done:
                break
        state = s_t
        info['goal_met'] = False
        info['cost'] = self.getCost(state)
        info['num_cv'] = 1 if info['cost'] >=0.5 else 0
        return state, reward, done, info

    def render(self, **args):
        return self._env.render(**args)

    def close(self):
        self._env.close()


def Env(env_name, seed, max_episode_length=1000, action_repeat=1):
    if 'walker' in env_name.lower():
        return WalkerEnv(env_name, seed, max_episode_length, action_repeat)
    elif 'cheetah' in env_name.lower():
        return CheetahEnv(env_name, seed, max_episode_length, action_repeat)
    elif 'jackal' in env_name.lower() or 'button' in env_name.lower():
        return gym.make(env_name)
    elif '4' in env_name.lower():
        return GymEnv4(env_name, seed, max_episode_length, action_repeat)
    else:
        return GymEnv(env_name, seed, max_episode_length, action_repeat)
