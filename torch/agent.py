from typing import Optional, List

from utils.color import cprint
from models import Policy
from models import Value2
from models import Value
import ctypes

from torch.distributions import Normal
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


class Agent:
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'

        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device, dtype=torch.float32)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device, dtype=torch.float32)
        self.n_envs = args.n_envs
        self.n_past_steps = args.n_steps if args.n_past_steps <= 0 else args.n_past_steps
        self.n_past_steps_per_env = int(self.n_past_steps/self.n_envs)
        self.n_update_steps = args.n_update_steps
        self.n_update_steps_per_env = int(self.n_update_steps/self.n_envs)

        # for RL
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.max_grad_norm = args.max_grad_norm
        self.gae_coeff = args.gae_coeff

        # for replay buffer
        self.replay_buffer_per_env = int(args.len_replay_buffer/args.n_envs)
        self.replay_buffer = [deque(maxlen=self.replay_buffer_per_env) for _ in range(args.n_envs)]

        # for trust region
        self.damping_coeff = args.damping_coeff
        self.num_conjugate = args.num_conjugate
        self.line_decay = args.line_decay
        self.max_kl = args.max_kl

        # for constraint
        self.cost_d = args.cost_d
        self.cost_alpha = args.cost_alpha
        self.sigma_unit = norm.pdf(norm.ppf(self.cost_alpha))/self.cost_alpha

        # declare networks
        self.policy = Policy(args).to(args.device)
        self.reward_value = Value(args).to(args.device)
        self.cost_value = Value(args).to(args.device)
        self.cost_std_value = Value2(args).to(args.device)
        self.cost_var_value = lambda x: torch.square(self.cost_std_value(x))

        # optimizers
        self.reward_value_optimizer = torch.optim.Adam(self.reward_value.parameters(), lr=self.lr)
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=self.lr)
        self.cost_std_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=self.lr)

        # load
        self._load()

    """ public functions
    """
    def getAction(self, state, is_train):
        mean, log_std, std = self.policy(state)
        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self._unnormalizeAction(mean + noise*std)
        else:
            action = self._unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)
        return action, clipped_action, mean, std

    def addTransition(self, env_idx, state, action, mu_mean, mu_std, reward, cost, done, fail, next_state):
        self.replay_buffer[env_idx].append([state, action, mu_mean, mu_std, reward, cost, done, fail, next_state])

    def train(self):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_list = []
        cost_var_targets_list = []
        reward_gaes_list = []
        cost_gaes_list = []
        cost_var_gaes_list = []
        mu_means_list = []
        mu_stds_list = []
        cost_mean = None
        cost_var_mean = None

        # latest trajectory
        temp_states_list, temp_actions_list, temp_reward_targets_list, temp_cost_targets_list, temp_cost_var_targets_list, \
            temp_reward_gaes_list, temp_cost_gaes_list, temp_cost_var_gaes_list, \
            temp_mu_means_list, temp_mu_stds_list, cost_mean, cost_var_mean = self._getTrainBatches(is_latest=True)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        cost_targets_list += temp_cost_targets_list
        cost_var_targets_list += temp_cost_var_targets_list
        reward_gaes_list += temp_reward_gaes_list
        cost_gaes_list += temp_cost_gaes_list
        cost_var_gaes_list += temp_cost_var_gaes_list
        mu_means_list += temp_mu_means_list
        mu_stds_list += temp_mu_stds_list

        # random trajectory
        temp_states_list, temp_actions_list, temp_reward_targets_list, temp_cost_targets_list, temp_cost_var_targets_list, \
            temp_reward_gaes_list, temp_cost_gaes_list, temp_cost_var_gaes_list, \
            temp_mu_means_list, temp_mu_stds_list, _, _ = self._getTrainBatches(is_latest=False)
        states_list += temp_states_list
        actions_list += temp_actions_list
        reward_targets_list += temp_reward_targets_list
        cost_targets_list += temp_cost_targets_list
        cost_var_targets_list += temp_cost_var_targets_list
        reward_gaes_list += temp_reward_gaes_list
        cost_gaes_list += temp_cost_gaes_list
        cost_var_gaes_list += temp_cost_var_gaes_list
        mu_means_list += temp_mu_means_list
        mu_stds_list += temp_mu_stds_list

        # convert to tensor
        with torch.no_grad():
            states_tensor = torch.tensor(np.concatenate(states_list, axis=0), device=self.device, dtype=torch.float32)
            actions_tensor = self._normalizeAction(torch.tensor(np.concatenate(actions_list, axis=0), device=self.device, dtype=torch.float32))
            reward_targets_tensor = torch.tensor(np.concatenate(reward_targets_list, axis=0), device=self.device, dtype=torch.float32)
            cost_targets_tensor = torch.tensor(np.concatenate(cost_targets_list, axis=0), device=self.device, dtype=torch.float32)
            cost_std_targets_tensor = torch.tensor(np.sqrt(np.concatenate(cost_var_targets_list, axis=0)), device=self.device, dtype=torch.float32)
            reward_gaes_tensor = torch.tensor(np.concatenate(reward_gaes_list, axis=0), device=self.device, dtype=torch.float32)
            cost_gaes_tensor = torch.tensor(np.concatenate(cost_gaes_list, axis=0), device=self.device, dtype=torch.float32)
            cost_var_gaes_tensor = torch.tensor(np.concatenate(cost_var_gaes_list, axis=0), device=self.device, dtype=torch.float32)
            mu_means_tensor = torch.tensor(np.concatenate(mu_means_list, axis=0), device=self.device, dtype=torch.float32)
            mu_stds_tensor = torch.tensor(np.concatenate(mu_stds_list, axis=0), device=self.device, dtype=torch.float32)

        # ================== Value Update ================== #
        for _ in range(self.n_epochs):
            reward_value_loss = torch.mean(0.5*torch.square(self.reward_value(states_tensor) - reward_targets_tensor))
            self.reward_value_optimizer.zero_grad()
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_value.parameters(), self.max_grad_norm)
            self.reward_value_optimizer.step()

            cost_value_loss = torch.mean(0.5*torch.square(self.cost_value(states_tensor) - cost_targets_tensor))
            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_value.parameters(), self.max_grad_norm)
            self.cost_value_optimizer.step()

            cost_var_value_loss = torch.mean(0.5*torch.square(self.cost_std_value(states_tensor) - cost_std_targets_tensor))
            self.cost_std_value_optimizer.zero_grad()
            cost_var_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cost_std_value.parameters(), self.max_grad_norm)
            self.cost_std_value_optimizer.step()
        # ================================================== #

        # ================= Policy Update ================= #
        # backup old policy
        means, _, stds = self.policy(states_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()
        cur_dists = Normal(means, stds)
        old_dists = Normal(old_means, old_stds)
        mu_dists = Normal(mu_means_tensor, mu_stds_tensor)
        kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
        kl_bonus = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(mu_dists, old_dists), dim=1))
        kl_bonus = torch.sqrt(kl_bonus*(self.max_kl + 0.25*kl_bonus)) - 0.5*kl_bonus
        max_kl = torch.clamp(self.max_kl - kl_bonus, 0.0, np.inf)

        # get objective & cost surrogate
        old_log_probs = torch.sum(old_dists.log_prob(actions_tensor), dim=1)
        mu_log_probs = torch.sum(mu_dists.log_prob(actions_tensor), dim=1)
        old_prob_ratios = torch.clamp(torch.exp(old_log_probs - mu_log_probs), 0.0, 1.0)
        objective, entropy = self._getObjective(cur_dists, old_dists, actions_tensor, reward_gaes_tensor, old_prob_ratios)
        cost_surrogate = self._getCostSurrogate(cur_dists, old_dists, actions_tensor, \
            cost_gaes_tensor, cost_var_gaes_tensor, old_prob_ratios, cost_mean, cost_var_mean)

        # get gradient
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        grad_b = flatGrad(-cost_surrogate, self.policy.parameters(), retain_graph=True)
        H_inv_g = self._conjugateGradient(kl, grad_g)
        approx_g = self._Hx(kl, H_inv_g)
        c_value = cost_surrogate.item() - self.cost_d/(1.0 - self.discount_factor)

        # ======== solve Lagrangian problem ======== #
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, H_inv_g)
            optim_case = 4
        else:
            H_inv_b = self._conjugateGradient(kl, grad_b)
            approx_b = self._Hx(kl, H_inv_b)
            scalar_q = torch.dot(approx_g, H_inv_g)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2/scalar_s
            B_value = 2*max_kl - c_value**2/scalar_s
            if c_value < 0 and B_value <= 0:
                optim_case = 3
            elif c_value < 0 and B_value > 0:
                optim_case = 2
            elif c_value >= 0 and B_value > 0:
                optim_case = 1
            else:
                optim_case = 0
        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q/(2*max_kl))
            nu = 0
        elif optim_case in [1, 2]:
            LA, LB = [0, scalar_r/c_value], [scalar_r/c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value/B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q/(2*max_kl)), LB)
            f_a = lambda lam : -0.5*(A_value/(lam + EPS) + B_value*lam) - scalar_r*c_value/(scalar_s + EPS)
            f_b = lambda lam : -0.5*(scalar_q/(lam + EPS) + 2*max_kl*lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam*c_value - scalar_r)/(scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2*max_kl/(scalar_s + EPS))
        # ========================================== #

        # line search
        with torch.no_grad():
            delta_theta = (1./(lam + EPS))*(H_inv_g + nu*H_inv_b) if optim_case > 0 else nu*H_inv_b
            beta = 1.0
            init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
            init_objective = objective.clone().detach()
            init_cost_surrogate = cost_surrogate.clone().detach()
            while True:
                theta = beta*delta_theta + init_theta
                self._applyParams(theta)
                means, _, stds = self.policy(states_tensor)
                cur_dists = Normal(means, stds)
                kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(old_dists, cur_dists), dim=1))
                objective, entropy = self._getObjective(cur_dists, old_dists, actions_tensor, reward_gaes_tensor, old_prob_ratios)
                cost_surrogate = self._getCostSurrogate(cur_dists, old_dists, actions_tensor, \
                    cost_gaes_tensor, cost_var_gaes_tensor, old_prob_ratios, cost_mean, cost_var_mean)
                if kl <= max_kl and (objective >= init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                    break
                beta *= self.line_decay
        # ================================================= #

        return objective.item(), cost_surrogate.item(), reward_value_loss.item(), cost_value_loss.item(), \
            cost_var_value_loss.item(), entropy.item(), kl.item(), optim_case

    def save(self):
        torch.save({
            'cost_value': self.cost_value.state_dict(),
            'cost_std_value': self.cost_std_value.state_dict(),
            'reward_value': self.reward_value.state_dict(),
            'policy': self.policy.state_dict(),
            'cost_value_optimizer': self.cost_value_optimizer.state_dict(),
            'cost_std_value_optimizer': self.cost_std_value_optimizer.state_dict(),
            'reward_value_optimizer': self.reward_value_optimizer.state_dict(),
        }, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    """ private functions
    """
    def _normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def _unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def _getObjective(self, cur_dists, old_dists, actions, reward_gaes, old_prob_ratios):
        entropy = torch.mean(torch.sum(cur_dists.entropy(), dim=1))
        cur_log_probs = torch.sum(cur_dists.log_prob(actions), dim=1)
        old_log_probs = torch.sum(old_dists.log_prob(actions), dim=1)
        prob_ratios = torch.exp(cur_log_probs - old_log_probs)
        reward_gaes_mean = torch.mean(reward_gaes*old_prob_ratios)
        reward_gaes_std = torch.std(reward_gaes*old_prob_ratios)
        objective = torch.mean(prob_ratios*(reward_gaes*old_prob_ratios - reward_gaes_mean)/(reward_gaes_std + EPS))
        return objective, entropy

    def _getCostSurrogate(self, cur_dists, old_dists, actions, cost_gaes, cost_var_gaes, old_prob_ratios, cost_mean, cost_var_mean):
        cur_log_probs = torch.sum(cur_dists.log_prob(actions), dim=1)
        old_log_probs = torch.sum(old_dists.log_prob(actions), dim=1)
        prob_ratios = torch.exp(cur_log_probs - old_log_probs)
        cost_gaes_mean = torch.mean(cost_gaes*old_prob_ratios)
        cost_var_gaes_mean = torch.mean(cost_var_gaes*old_prob_ratios)
        approx_cost_mean = cost_mean + (1.0/(1.0 - self.discount_factor))*torch.mean(prob_ratios*(cost_gaes*old_prob_ratios - cost_gaes_mean))
        approx_cost_var = cost_var_mean + (1.0/(1.0 - self.discount_factor**2))*torch.mean(prob_ratios*(cost_var_gaes*old_prob_ratios - cost_var_gaes_mean))
        cost_surrogate = approx_cost_mean + self.sigma_unit*torch.sqrt(torch.clamp(approx_cost_var, EPS, np.inf))
        return cost_surrogate

    def _getGaesTargets(self, rewards, values, dones, fails, next_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = rewards[t] + self.discount_factor*(1.0 - fails[t])*next_values[t] \
                            + self.discount_factor*(1.0 - dones[t])*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - values[t])
        gaes = targets - values
        return gaes, targets

    def _getVarGaesTargets(self, rewards, values, var_values, dones, fails, next_values, next_var_values, rhos):
        delta = 0.0
        targets = np.zeros_like(rewards)
        for t in reversed(range(len(targets))):
            targets[t] = np.square(rewards[t] + (1.0 - fails[t])*self.discount_factor*next_values[t]) - np.square(values[t]) + \
                            (1.0 - fails[t])*(self.discount_factor**2)*next_var_values[t] + \
                            (1.0 - dones[t])*(self.discount_factor**2)*delta
            delta = self.gae_coeff*rhos[t]*(targets[t] - var_values[t])
        gaes = targets - var_values
        targets = np.clip(targets, 0.0, np.inf)
        return gaes, targets

    def _getTrainBatches(self, is_latest=False):
        states_list = []
        actions_list = []
        reward_targets_list = []
        cost_targets_list = []
        cost_var_targets_list = []
        reward_gaes_list = []
        cost_gaes_list = []
        cost_var_gaes_list = []
        cost_mean_list = []
        cost_var_mean_list = []
        mu_means_list = []
        mu_stds_list = []

        with torch.no_grad():
            for env_idx in range(self.n_envs):
                n_latest_steps = min(len(self.replay_buffer[env_idx]), self.n_past_steps_per_env)
                if is_latest:
                    start_idx = len(self.replay_buffer[env_idx]) - n_latest_steps
                    end_idx = start_idx + n_latest_steps
                    env_trajs = list(self.replay_buffer[env_idx])[start_idx:end_idx]
                else:
                    n_update_steps = min(len(self.replay_buffer[env_idx]), self.n_update_steps_per_env)
                    if n_update_steps <= n_latest_steps:
                        continue
                    start_idx = np.random.randint(0, len(self.replay_buffer[env_idx]) - n_update_steps + 1)
                    end_idx = start_idx + (n_update_steps - n_latest_steps)
                    env_trajs = list(self.replay_buffer[env_idx])[start_idx:end_idx]

                states = np.array([traj[0] for traj in env_trajs])
                actions = np.array([traj[1] for traj in env_trajs])
                mu_means = np.array([traj[2] for traj in env_trajs])
                mu_stds = np.array([traj[3] for traj in env_trajs])
                rewards = np.array([traj[4] for traj in env_trajs])
                costs = np.array([traj[5] for traj in env_trajs])
                dones = np.array([traj[6] for traj in env_trajs])
                fails = np.array([traj[7] for traj in env_trajs])
                next_states = np.array([traj[8] for traj in env_trajs])

                # convert to tensor
                states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
                actions_tensor = self._normalizeAction(torch.tensor(actions, device=self.device, dtype=torch.float32))
                next_states_tensor = torch.tensor(next_states, device=self.device, dtype=torch.float32)
                mu_means_tensor = torch.tensor(mu_means, device=self.device, dtype=torch.float32)
                mu_stds_tensor = torch.tensor(mu_stds, device=self.device, dtype=torch.float32)

                # for rho
                means_tensor, _, stds_tensor = self.policy(states_tensor)
                old_dists = torch.distributions.Normal(means_tensor, stds_tensor)
                mu_dists = torch.distributions.Normal(mu_means_tensor, mu_stds_tensor)
                old_log_probs_tensor = torch.sum(old_dists.log_prob(actions_tensor), dim=1)
                mu_log_probs_tensor = torch.sum(mu_dists.log_prob(actions_tensor), dim=1)
                rhos_tensor = torch.clamp(torch.exp(old_log_probs_tensor - mu_log_probs_tensor), 0.0, 1.0)
                rhos = rhos_tensor.detach().cpu().numpy()

                # get GAEs and Tagets
                # for reward
                reward_values_tensor = self.reward_value(states_tensor)
                next_reward_values_tensor = self.reward_value(next_states_tensor)
                reward_values = reward_values_tensor.detach().cpu().numpy()
                next_reward_values = next_reward_values_tensor.detach().cpu().numpy()
                reward_gaes, reward_targets = self._getGaesTargets(rewards, reward_values, dones, fails, next_reward_values, rhos)
                # for cost
                cost_values_tensor = self.cost_value(states_tensor)
                next_cost_values_tensor = self.cost_value(next_states_tensor)
                cost_values = cost_values_tensor.detach().cpu().numpy()
                next_cost_values = next_cost_values_tensor.detach().cpu().numpy()
                cost_gaes, cost_targets = self._getGaesTargets(costs, cost_values, dones, fails, next_cost_values, rhos)
                # for cost var
                cost_var_values_tensor = self.cost_var_value(states_tensor)
                next_cost_var_values_tensor = self.cost_var_value(next_states_tensor)
                cost_var_values = cost_var_values_tensor.detach().cpu().numpy()
                next_cost_var_values = next_cost_var_values_tensor.detach().cpu().numpy()
                cost_square_gaes, cost_var_targets = self._getVarGaesTargets(costs, cost_values, cost_var_values, dones, fails, next_cost_values, next_cost_var_values, rhos)

                # add cost mean & cost variance mean
                cost_mean_list.append(np.mean(costs)/(1.0 - self.discount_factor))
                cost_var_mean_list.append(np.mean(cost_var_targets))

                # save
                states_list.append(states)
                actions_list.append(actions)
                reward_gaes_list.append(reward_gaes)
                cost_gaes_list.append(cost_gaes)
                cost_var_gaes_list.append(cost_square_gaes)
                reward_targets_list.append(reward_targets)
                cost_targets_list.append(cost_targets)
                cost_var_targets_list.append(cost_var_targets)
                mu_means_list.append(mu_means)
                mu_stds_list.append(mu_stds)

        # get cost mean & cost variance mean
        cost_mean = np.mean(cost_mean_list)
        cost_var_mean = np.mean(cost_var_mean_list)

        return states_list, actions_list, reward_targets_list, cost_targets_list, cost_var_targets_list, \
            reward_gaes_list, cost_gaes_list, cost_var_gaes_list, mu_means_list, mu_stds_list, cost_mean, cost_var_mean

    def _applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def _Hx(self, kl:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        '''
        get (Hessian of KL * x).
        input:
            kl: tensor(,)
            x: tensor(dim,)
        output:
            Hx: tensor(dim,)
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x*self.damping_coeff

    def _conjugateGradient(self, kl:torch.Tensor, g:torch.Tensor) -> torch.Tensor:
        '''
        get (H^{-1} * g).
        input:
            kl: tensor(,)
            g: tensor(dim,)
        output:
            H^{-1}g: tensor(dim,)
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r*r)
        for i in range(self.num_conjugate):
            Ap = self._Hx(kl, p)
            pAp = torch.sum(p*Ap)
            alpha = rs_old/(pAp + EPS)
            x += alpha*p
            r -= alpha*Ap
            rs_new = torch.sum(r*r)
            p = r + (rs_new/(rs_old + EPS))*p
            rs_old = rs_new
        return x

    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.reward_value.load_state_dict(checkpoint['reward_value'])
            self.cost_value.load_state_dict(checkpoint['cost_value'])
            self.cost_std_value.load_state_dict(checkpoint['cost_std_value'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.reward_value_optimizer.load_state_dict(checkpoint['reward_value_optimizer'])
            self.cost_value_optimizer.load_state_dict(checkpoint['cost_value_optimizer'])
            self.cost_std_value_optimizer.load_state_dict(checkpoint['cost_std_value_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            self.reward_value.initialize()
            self.cost_value.initialize()
            self.cost_std_value.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
