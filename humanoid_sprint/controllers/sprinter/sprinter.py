from controller import Robot, Motion, Supervisor
from collections import deque
from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import math
import os
import copy

STATE_DIM = 18
ACTION_DIM = 100
TRAIN_EPISODES=100
NUM_EPISODES =100
NUM_BC_EPISODES = 100
SEQUENCE_LENGTH=35
INITIAL_STATE=[2.813063620922607e-07, -2.8354001787624697e-07, 2.661361167672216e-07, 2.680192760864757e-07, 2.967163694572752e-07, 2.9883194662671264e-07, 2.4820736847423014e-07, 2.4949180207911914e-07, 1.9713857411518186e-07, 1.9756885276656546e-07, -1.866298555697199e-07, 1.8715301552270632e-07, [-5.021522061737612,  0.9999547666320551, 0.7058214811932133]]
ACTOR_LR=1e-4
CRITIC_LR=1e-4
CKPTPATH="ddpg_checkpoint.pth"
CKPTLPATH="ddpg_checkpoint_false.pth"
INITIAL_ACTIONS = [
           [-0.034,-0.518,1.047,-0.529,0.034,-0.034,-0.518,1.047,-0.529,0.034,],  
           [-0.044,-0.516,1.045,-0.53,0.044,-0.044,-0.516,1.045,-0.53,0.044,],
           [-0.063,-0.51,1.041,-0.531,0.063,-0.063,-0.51,1.041,-0.531,0.063,],
           [-0.08,-0.505,1.036,-0.531,0.08,-0.08,-0.505,1.036,-0.531,0.08,],
           [-0.101,-0.498,1.029,-0.531,0.101,-0.101,-0.498,1.029,-0.531,0.101,],
           [-0.136,-0.482,1.013,-0.531,0.136,-0.136,-0.482,1.013,-0.531,0.136],
           [-0.157,-0.468,0.999,-0.531,0.157,-0.157,-0.468,0.999,-0.531,0.157,],
           [-0.173,-0.454,0.987,-0.533,0.173,-0.174,-0.459,0.996,-0.537,0.174,],
           [-0.15,-0.44,0.984,-0.544,0.19,-0.197,-0.52,1.119,-0.599,0.197,],
           [-0.144,-0.442,0.999,-0.557,0.196,-0.211,-0.605,1.244,-0.639,0.211,],
           [-0.146,-0.452,1.033,-0.582,0.198,-0.223,-0.743,1.374,-0.631,0.223,],
           [-0.142,-0.454,1.051,-0.597,0.195,-0.219,-0.803,1.378,-0.575,0.219,],
           [-0.134,-0.447,1.056,-0.609,0.187,-0.205,-0.815,1.313,-0.498,0.205,],
           [-0.121,-0.427,1.045,-0.618,0.174,-0.184,-0.777,1.202,-0.425,0.184],
           [-0.128,-0.379,1.011,-0.632,0.145,-0.147,-0.678,1.037,-0.36,0.147,],
           [-0.119,-0.344,0.99,-0.646,0.119,-0.119,-0.634,1,-0.366,0.119,],
           [-0.065,-0.293,0.97,-0.677,0.065,-0.065,-0.604,1.022,-0.417,0.065,],
           [-0.019,-0.253,0.948,-0.696,0.019,-0.019,-0.579,1.033,-0.453,0.019,],
           [0.029,-0.204,0.913,-0.709,-0.029,0.029,-0.547,1.033,-0.486,-0.029,],
           [0.094,-0.124,0.838,-0.714,-0.094,0.094,-0.489,1.014,-0.525,-0.094,],
           [0.126,-0.096,0.828,-0.733,-0.126,0.12,-0.451,0.993,-0.542,-0.125,],
           [0.153,-0.142,0.934,-0.792,-0.153,0.108,-0.424,0.983,-0.559,-0.147,],
           [0.186,-0.337,1.196,-0.859,-0.186,0.117,-0.412,0.99,-0.578,-0.169],
           [0.199,-0.508,1.335,-0.827,-0.199,0.124,-0.42,1.008,-0.589,-0.177,],
           [0.201,-0.661,1.384,-0.722,-0.201,0.126,-0.432,1.03,-0.599,-0.178,],
           [0.184,-0.763,1.278,-0.515,-0.184,0.117,-0.435,1.047,-0.612,-0.17,],
           [0.164,-0.739,1.148,-0.409,-0.164,0.105,-0.418,1.038,-0.62,-0.157],
           [0.14,-0.683,1.039,-0.356,-0.14,0.122,-0.387,1.017,-0.63,-0.139,],
           [0.099,-0.628,1.006,-0.378,-0.099,0.099,-0.334,0.988,-0.654,-0.099,],
           [0.062,-0.608,1.022,-0.414,-0.062,0.062,-0.299,0.974,-0.675,-0.062,],
           [-0.007,-0.566,1.034,-0.468,0.007,-0.007,-0.233,0.935,-0.702,0.007,],
           [-0.055,-0.53,1.029,-0.499,0.055,-0.055,-0.181,0.892,-0.712,0.055,],
           [-0.095,-0.49,1.014,-0.524,0.095,-0.095,-0.126,0.839,-0.714,0.095,],
           [-0.121,-0.452,0.993,-0.541,0.126,-0.127,-0.097,0.83,-0.732,0.127,],
           [-0.105,-0.417,0.982,-0.565,0.157,-0.166,-0.194,1.018,-0.824,0.166,]                    
        ]



class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done, next_action=None):

        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            return x

        s = to_list(state)
        a = to_list(action)
        ns = to_list(next_state)
        na = to_list(next_action) if next_action is not None else None
        r = float(reward) if not isinstance(reward, torch.Tensor) else reward.item()
        d = bool(done)    if not isinstance(done,    torch.Tensor) else bool(done.item())

        self.buffer.append((s, a, r, ns, d, na))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones, next_actions = zip(*batch)

        states      = torch.tensor(states,      dtype=torch.float32)
        actions     = torch.tensor(actions,     dtype=torch.float32)
        rewards     = torch.tensor(rewards,     dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones,       dtype=torch.float32).unsqueeze(-1)

        if next_actions[0] is not None:
            next_actions = torch.tensor(next_actions, dtype=torch.float32)
        else:
            next_actions = None

        return states, actions, rewards, next_states, dones, next_actions

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class SequenceBCBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def push_sequence(self, states, actions):

        if isinstance(states, list) and isinstance(states[0], torch.Tensor):
            states = [s.detach().cpu().numpy().tolist() for s in states]
        if isinstance(actions, list) and isinstance(actions[0], torch.Tensor):
            actions = [a.detach().cpu().numpy().tolist() for a in actions]

        self.buffer.append((states, actions))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_sequences(self, batch_size):

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def flatten_sample(self, batch_size):

        data = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        all_states = []
        all_actions = []
        for s_seq, a_seq in data:
            all_states.extend(s_seq)
            all_actions.extend(a_seq)
        states = torch.tensor(all_states, dtype=torch.float32)
        actions = torch.tensor(all_actions, dtype=torch.float32)
        return states, actions

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        layers = []
        inp = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        layers.append(nn.Linear(inp, action_dim))
        self.net = nn.Sequential(*layers)

        self.register_buffer("joint_min", torch.tensor([
            -0.379, -1.774, -0.092, -1.189, -0.397,
            -0.738, -1.774, -0.092, -1.186, -0.769
        ], dtype=torch.float32))
        self.register_buffer("joint_max", torch.tensor([
            0.790, 0.484, 2.113, 0.923, 0.769,
            0.450, 0.484, 2.113, 0.932, 0.38
        ], dtype=torch.float32))
        
    def forward(self, state):

        if isinstance(state, (list, tuple, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        raw = self.net(state)

        action = torch.max(torch.min(raw, self.joint_max), self.joint_min)
        return action
        

        
        

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.l1  = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.l2  = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.l3  = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
      
        if state.dim()  == 1: state  = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)
        x = F.relu(self.bn1(self.l1(state)))          # [B, hidden_dim]
        x = torch.cat([x, action], dim=-1)            # [B, hidden_dim+action_dim]
        x = F.relu(self.bn2(self.l2(x)))              # [B, hidden_dim]
        q = self.l3(x)                                # [B, 1]
        return q



class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim,
                 actor_lr, critic_lr, device,
                 gamma=0.98, tau=1e-4):
        
        self.device = device
        self.gamma = gamma
        self.tau   = tau
        self.actor  = Actor(state_dim, 256, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.device = device 
        self.batch_size = 3
        self.min_buffer_size =3
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.bc_buffer = SequenceBCBuffer(capacity=10000)

        self._train_logs = {
            "rewards": [], "target_qs": [], "current_qs": [],
            "actor_losses": [], "critic_losses": [], "q_vs_r_pairs": [] 
        }
        self.demo_actions=torch.tensor(INITIAL_ACTIONS)
        self.actor_update_start = 0

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
    
    def freeze_critic(self):
        for param in self.critic.parameters():
            param.requires_grad = False
        print(" Critic network has been frozen.")
    
    def unfreeze_critic(self):
        for param in self.critic.parameters():
            param.requires_grad = True
        print(" Critic network has been unfrozen.")

    def train(self, critic_updates: int = 2):
        if len(self.replay_buffer) < self.batch_size:
            print(" Not enough data for training.")
            return 0.0, 0.0
    
        states, actions, rewards, next_states, dones, _ = \
            self.replay_buffer.sample_batch(self.batch_size)
    
        device = self.device
        states      = states.to(device)
        actions     = actions.to(device)
        rewards     = rewards.to(device)
        next_states = next_states.to(device)
        dones       = dones.to(device)
    
        avg_critic_loss = 0.0
        current_q_val = 0.0
        target_q_val  = 0.0
    
  
        if any(p.requires_grad for p in self.critic.parameters()):
            total_critic_loss = 0.0
            for _ in range(critic_updates):
                with torch.no_grad():
                    next_a   = self.actor_target(next_states)
                    next_q   = self.critic_target(next_states, next_a)
                    target_q = rewards + self.gamma * (1 - dones) * next_q
    
                current_q   = self.critic(states, actions)
                critic_loss = F.mse_loss(current_q, target_q)
    
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.soft_update(self.critic, self.critic_target)
    
                total_critic_loss += critic_loss.item()
    
            avg_critic_loss = total_critic_loss / critic_updates
            current_q_val = current_q.mean().item()
            target_q_val  = target_q.mean().item()
        else:
            print(" Critic is frozen, skipping Critic update")
    

        pred_a     = self.actor(states)
        actor_loss = - self.critic(states, pred_a).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target)
    
        actor_loss_val = actor_loss.item()
        reward_val = rewards.mean().item()
    

        self._train_logs.setdefault("actor_losses",  []).append(actor_loss_val)
        self._train_logs.setdefault("critic_losses", []).append(avg_critic_loss)
        self._train_logs.setdefault("current_qs",     []).append(current_q_val)
        self._train_logs.setdefault("target_qs",      []).append(target_q_val)
        self._train_logs.setdefault("rewards",        []).append(reward_val)
    
        print(f" [Train] Actor loss={actor_loss_val:.4f}, "
              f"Critic loss={avg_critic_loss:.4f}, "
              f"Q mean={current_q_val:.3f}→{target_q_val:.3f}, "
              f"Reward={reward_val:.3f}")
    
        return actor_loss_val, avg_critic_loss


    




    
    
    def demo_train(self, repeat: int = 1):
        if len(self.rl_buffer) < self.batch_size:
            print(" Not enough data for demo training.")
            return 0.0, 0.0
    
       
        states, actions, rewards, next_states, dones, next_actions, mask = \
            self.rl_buffer.sequence_batch_sample(self.batch_size)
    
        B, T, _ = states.shape
        f_mask = mask.reshape(B * T, 1)
        total_critic_loss = 0.0
    
        for r in range(repeat):
            flat = lambda x: x.reshape(B * T, -1)
            f_states = flat(states)
            f_actions = flat(actions)
            f_next_states = flat(next_states)
            f_next_actions = flat(next_actions)
            f_rewards = rewards.reshape(B * T, 1)
            f_dones = dones.reshape(B * T, 1)
    
            with torch.no_grad():
                next_q = self.critic_target(f_next_states, f_next_actions)
                target_q = f_rewards + self.gamma * (1 - f_dones) * next_q
    
            current_q = self.critic(f_states, f_actions)
   
            td_error = F.smooth_l1_loss(current_q, target_q, reduction='none') * f_mask
            critic_loss = td_error.sum() / f_mask.sum().clamp(min=1.0)
    
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()
    
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
    
            if r == repeat - 1:
                q_vs_r = list(zip(current_q.squeeze().tolist(), f_rewards.squeeze().tolist()))
                self._train_logs.setdefault("q_vs_r_pairs", []).append(q_vs_r)
    
        self._train_logs["rewards"].append(f_rewards.mean().item())
        self._train_logs["target_qs"].append(target_q.mean().item())
        self._train_logs["current_qs"].append(current_q.mean().item())
        self._train_logs["actor_losses"].append(0.0)
        self._train_logs["critic_losses"].append(total_critic_loss / repeat)
    
        print(f"[Demo Train] reward={f_rewards.mean():.3f}, current_q={current_q.mean():.3f}, "
              f"target_q={target_q.mean():.3f}, critic_loss={critic_loss.item():.4f}")
    
        return 0.0, total_critic_loss / repeat

    

    
    def train_bc(self,
                      epochs: int = 200,
                      lr: float = 5e-5,
                      weight_decay: float = 0,
                      clip_norm: float = None,
                      patience: int = 20):
        """
          1) flatten_sample -> states[N,dim], actions[N,dim]
          2) pred = actor(states)
          3) loss = MSE(pred, actions)
        """
        optimizer = torch.optim.Adam(
            self.actor.net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        best_loss = float('inf')
        no_improve = 0
    
        for epoch in range(1, epochs+1):
       
            states, actions = self.bc_buffer.flatten_sample(self.batch_size)
            states  = states.to(self.device)
            actions = actions.to(self.device)
    

            preds = self.actor(states)       
            loss  = loss_fn(preds, actions)
    
            optimizer.zero_grad()
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.net.parameters(), clip_norm
                )
            optimizer.step()
    
            avg = loss.item()
            print(f"Epoch [{epoch}/{epochs}] flat‐mapped-MSE: {avg:.6f}")
    
            # Early stopping
            if avg + 1e-6 < best_loss:
                best_loss  = avg
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f" Early stopping at epoch {epoch}")
                    break
    
        print("=== Flat‐BC Training Finished ===")


    




    

    
            
    def save_checkpoint(self, filepath="ddpg_checkpoint.pth"):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "train_logs": self._train_logs,
        }
        torch.save(checkpoint, filepath)
        print(f"Saved DDPG checkpoint (with target networks) to {filepath}")

    
    def load_checkpoint(self, filepath="ddpg_checkpoint.pth", actor_lr: float = 1e-4, critic_lr: float = 1e-4):
        import os, torch
        if not os.path.exists(filepath):
            print(f" Checkpoint not found at {filepath}")
            return
    
        checkpoint = torch.load(filepath, map_location="cpu")

        name, _ = next(self.critic.named_parameters())
        print(f" Before load: critic parameter '{name}' first 5 vals =",
              self.critic.state_dict()[name].view(-1)[:5])

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor.to(self.device)
        self.critic.to(self.device)
    

        if "actor_target_state_dict" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
            self.actor_target.to(self.device)
            print(" Loaded actor_target from checkpoint.")
        else:
            print("actor_target not found in checkpoint.")
    
        if "critic_target_state_dict" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_target.to(self.device)
            print(" Loaded critic_target from checkpoint.")
        else:
            print(" critic_target not found in checkpoint.")
    
        print(f"▶ After load: critic parameter '{name}' first 5 vals =",
              self.critic.state_dict()[name].view(-1)[:5])
        print(f"▶ After load: critic_target parameter '{name}' first 5 vals =",
              self.critic_target.state_dict()[name].view(-1)[:5])

        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    
        for opt in (self.actor_optimizer, self.critic_optimizer):
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
    
        if actor_lr is not None:
            for pg in self.actor_optimizer.param_groups:
                pg['lr'] = actor_lr
            print(f" Actor optimizer learning rate set to {actor_lr}")
        if critic_lr is not None:
            for pg in self.critic_optimizer.param_groups:
                pg['lr'] = critic_lr
            print(f" Critic optimizer learning rate set to {critic_lr}")
    
        print(f" Loaded checkpoint from {filepath} onto {self.device}")



    def load_bc(self, filepath="actor_bc.pth"):
        import os
        if not os.path.exists(filepath):
            print(f" Behavior cloning model not found at {filepath}")
            return
    
        state_dict = torch.load(filepath, map_location="cpu")
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)  # ✅ 同步 target actor
        self.actor.to(self.device)
        self.actor_target.to(self.device)
    
        print(f"Loaded behavior cloning model into actor and target actor from '{filepath}'")


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state)
        return action.squeeze(0)
        

    def log_tensor_stats(name, tensor):
        if isinstance(tensor, torch.Tensor):
            print(f" {name} - shape: {tensor.shape}, mean: {tensor.mean().item():.4e}, max: {tensor.abs().max().item():.4e}, min: {tensor.min().item():.4e}")
            if torch.isnan(tensor).any():
                print(f" NaN detected in {name}")
            if torch.isinf(tensor).any():
                print(f" Inf detected in {name}")
    
    def log_param_grad(model):
        print("==== [Param Gradients] ====")
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = param.grad
                if grad is not None:
                    print(f"Layer {name} → grad std: {grad.std().item():.4e}, mean: {grad.mean().item():.4e}, |param| max: {param.abs().max().item():.4e}")
                    if torch.isnan(grad).any():
                        print(f" NaN in grad of {name}")
                    if torch.isinf(grad).any():
                        print(f" Inf in grad of {name}")
        print("===========================")
    
    def log_model_output(model, state, action, pre_fc3_layer=None):
        with torch.no_grad():
            if pre_fc3_layer is not None:
                x = model.leaky_relu(model.ln1(model.fc1(torch.cat([state, action], dim=-1))))
                x = model.leaky_relu(model.ln2(model.fc2(x)))
                log_tensor_stats("pre_fc3 Q", x)
            q = model(state, action)
            log_tensor_stats("Q value", q)
    
    def log_buffer_stats(states, actions, rewards, next_states, dones):
        log_tensor_stats("states", states)
        log_tensor_stats("actions", actions)
        log_tensor_stats("rewards", rewards)
        log_tensor_stats("next_states", next_states)
        log_tensor_stats("dones", dones)
    
    def log_train_step(ddpg, states, actions, rewards, next_states, dones):
        print("\n [Training Step Diagnostic Log]")
        log_buffer_stats(states, actions, rewards, next_states, dones)
    
        with torch.no_grad():
            next_actions = ddpg.actor(next_states)
            target_q = ddpg.critic(next_states, next_actions)
            current_q = ddpg.critic(states, actions)
    
            log_tensor_stats("Target Q", target_q)
            log_tensor_stats("Current Q", current_q)
            log_tensor_stats("Predicted Action", ddpg.actor(states))
    
            log_model_output(ddpg.critic, states, actions, pre_fc3_layer=True)
    
        critic_loss = F.mse_loss(current_q, rewards + ddpg.gamma * (1 - dones) * target_q)
        print(f"Loss Critic: {critic_loss.item():.4e}")
        print("==============================\n")

    def _plot_training_summary(self, save_path=None): 
        logs = self._train_logs
        steps = list(range(len(logs.get("actor_losses", []))))
    
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(" Training Summary", fontsize=16)
    
        # 0,0: Actor / Critic Loss
        axs[0, 0].plot(steps, logs.get("actor_losses", []), label="Actor Loss")
        axs[0, 0].plot(steps, logs.get("critic_losses", []), label="Critic Loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].legend()
    
        # 0,1: Q values
        axs[0, 1].plot(steps, logs.get("current_qs", []), label="Current Q")
        axs[0, 1].plot(steps, logs.get("target_qs", []), label="Target Q")
        axs[0, 1].set_title("Q-values")
        axs[0, 1].legend()
    
        # 1,0: Reward per Train
        axs[1, 0].plot(steps, logs.get("rewards", []), label="Reward")
        axs[1, 0].set_title("Reward per Train")
        axs[1, 0].legend()
    
        # 1,1: Critic grad norm, TD-error mean, mask ratio
        if "critic_grad_norms" in logs:
            axs[1, 1].plot(steps[:len(logs["critic_grad_norms"])],
                           logs["critic_grad_norms"], label="Critic ∥∇Q∥")
        if "td_error_means" in logs:
            axs[1, 1].plot(steps[:len(logs["td_error_means"])],
                           logs["td_error_means"], label="TD-error mean")
        if "mask_ratios" in logs:
            axs[1, 1].plot(steps[:len(logs["mask_ratios"])],
                           logs["mask_ratios"], label="Mask ratio")
        axs[1, 1].set_title("Critic Diagnostics")
        axs[1, 1].legend()
    
        # 2,0: Actor grad norm & action distrib
        have_actor_grad = "actor_grad_norms" in logs
        have_action_means = "action_means" in logs
        have_action_stds = "action_stds" in logs
        plotted = False
        if have_actor_grad:
            axs[2, 0].plot(steps[:len(logs["actor_grad_norms"])],
                           logs["actor_grad_norms"],     label="Actor ∥∇π∥")
            plotted = True
        if have_action_means:
            axs[2, 0].plot(steps[:len(logs["action_means"])],
                           logs["action_means"],         label="Action mean")
            plotted = True
        if have_action_stds:
            axs[2, 0].plot(steps[:len(logs["action_stds"])],
                           logs["action_stds"],          label="Action std")
            plotted = True
        if plotted:
            axs[2, 0].set_title("Actor Diagnostics")
            axs[2, 0].legend()
    
        # 2,1: Q vs Reward scatter (last batch)
        if "q_vs_r_pairs" in logs and logs["q_vs_r_pairs"]:
            q_vs_r = logs["q_vs_r_pairs"][-1]
            q_vals, r_vals = zip(*q_vs_r)
            axs[2, 1].scatter(r_vals, q_vals, alpha=0.3)
            axs[2, 1].set_xlabel("Reward")
            axs[2, 1].set_ylabel("Q")
            axs[2, 1].set_title("Q vs Reward (Last Batch)")
    
        # Print Critic Q and Actual Reward for Debugging
        if "critic_qs" in logs and "rewards" in logs:
            axs[2, 1].scatter(logs["rewards"], logs["critic_qs"], alpha=0.3, label="Critic Q vs Reward")
            axs[2, 1].legend()
    
        for ax in axs.flat:
            ax.grid(True)
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        if save_path:
            plt.savefig(save_path)
            print(f" Training summary saved to {save_path}")
            plt.show()
        else:
            plt.show()
    
    
    

         

def print_model_weights(self,model, name="Model"):
    print(f"\n {name} weights summary:")
    for n, p in model.named_parameters():
        if p is not None:
            data = p.data
            print(f"   {n}: shape={tuple(data.shape)}, mean={data.mean():.4f}, std={data.std():.4f}, min={data.min():.4f}, max={data.max():.4f}")
            if torch.isnan(data).any():
                print(f"   {n} contains NaNs!")
            if torch.isinf(data).any():
                print(f"   {n} contains Infs!")

       
class Sprinter(Supervisor):
    def __init__(self):
        super().__init__()

        self.timeStep = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # —— 把 device 传给 DDPG —— 
        self.ddpg = DDPG(
            state_dim=STATE_DIM,
            action_dim=10,
            hidden_dim=512,
            actor_lr=1e-4,
            critic_lr=1e-4,
            device=self.device,        
            gamma=0.98,
        )
        # （此时 DDPG.__init__ 里已经完成 .to(self.device)）

    def initialize(self):
        self.timeStep = 40
        self.RShoulderPitch = self.getDevice('RShoulderPitch')
        self.LShoulderPitch = self.getDevice('LShoulderPitch')
        self.RShoulderPitch.setPosition(1.1)
        self.LShoulderPitch.setPosition(1.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.joint_names = [
            'RHipYawPitch', 'LHipYawPitch',
            'RHipRoll', 'LHipRoll',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
        self.motors = {name: self.getDevice(name) for name in self.joint_names}
        self.sensors = {}
        for name in self.joint_names:
            sensor_name = name + "S"
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.timeStep)
            self.sensors[sensor_name] = sensor
            setattr(self, sensor_name.replace("S", "Sensor"), sensor)

        self.accelerometer = self.getDevice("accelerometer")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.accelerometer.enable(self.timeStep)
        self.imu.enable(self.timeStep)
        self.gyro.enable(self.timeStep)

        for _ in range(5):
            self.step(self.timeStep)

        self.nao = self.getFromDef("NAO")
        self.translation = self.nao.getField("translation")
        self.rotation = self.nao.getField("rotation")



        self.initVector = self.get_translation()
        print("self.initVector", self.initVector)
        self.initYaw =0
        print("self.initYaw", self.initYaw)
        self.pre_position = self.get_translation()
        self.start_position = self.get_translation()
        self.current_position = self.get_translation()
        self.pre_yaw = self.get_yaw()
        self.start_yaw = self.get_yaw()
        self.current_yaw = self.get_yaw()
        self.steps =[]
        # Y-axis data: generic series a and b
        self.a_values = []  # e.g. move rewards, or any first series
        self.b_values = []  # e.g. positions, or any second series

        self.state_logs=[]
        self.columns = [
            "phase", "roll", "yaw", "gyro_x", "gyro_y", "cm_norm",
            "RHipYawPitch","LHipYawPitch","RHipRoll","LHipRoll",
            "RHipPitch","LHipPitch","RKneePitch","LKneePitch",
            "RAnklePitch","LAnklePitch","RAnkleRoll","LAnkleRoll"
        ]
    def run_bc_once(self,
                    collect_eps: int = 40,
                    bc_epochs: int = 100,
                    bc_lr: float = 1e-3,
                    bc_model_path: str = "actor_bc.pth"):
        """
        1) 清空旧 BC 缓存，重新采集示范数据
        2) 调用 train_bc 打印并返回 loss 列表
        3) 用固定文件名保存训练好的 Actor
        """
        #self.ddpg.load_bc()
        print("\n🔄 开始 BC 数据采集 …")
        self.ddpg.bc_buffer.clear()
        self.train_bc_data_collection(num_episodes=collect_eps)

        print("\n📦 开始 BC 预训练 …")
        losses = self.ddpg.train_bc(epochs=bc_epochs, lr=bc_lr)

        torch.save(self.ddpg.actor.state_dict(), bc_model_path)
        print(f"✅ 已保存 BC 训练后的 Actor 到 {bc_model_path}")

        return losses

        print("\n🎉 多轮 BC 预训练完成！")

    
    
    def print_model_weights(self, model, name="Model"):
        print(f"\n🎯 {name} weights summary:")
        for n, p in model.named_parameters():
            if p is not None:
                data = p.data
                print(f"  📌 {n}: shape={tuple(data.shape)}, mean={data.mean().item():.4f}, std={data.std().item():.4f}, min={data.min().item():.4f}, max={data.max().item():.4f}")
                if torch.isnan(data).any():
                    print(f"  ❌ {n} contains NaNs!")
                if torch.isinf(data).any():
                    print(f"  ❌ {n} contains Infs!")

    
    def print_joint_ranges(self):
        for name in self.joint_names:
            motor = self.getDevice(name)
            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            print(f"{name}: Min = {min_pos:.3f}, Max = {max_pos:.3f}")
  
    def train_bc_data_collection(self, num_episodes):
        """
        专门用于采集 BC 数据（基于 SequenceReplayBuffer）
        """
        for episode in range(num_episodes):
            self.pre_position = self.get_translation()
            print(f"[BC Collection] Episode: {episode}")
    
            state_seq, action_seq = [], []
            current_state = self.get_current_state(0)
    
            for s in range(SEQUENCE_LENGTH):
                action = torch.tensor(INITIAL_ACTIONS[s % len(INITIAL_ACTIONS)], dtype=torch.float32)
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                next_state = self.get_current_state(s)

                state_seq.append(current_state)
                action_seq.append(action)
    
                # 提前中止逻辑
                pm = self.get_plane_movement()
                if self.get_center_of_mass() < 0.6:
                    print("falling")
                    break
                elif abs(pm["total_angle"]) > 20:
                    print("wrong direction")
                    break
    
                current_state = next_state
    
            if len(state_seq) == SEQUENCE_LENGTH:
                self.ddpg.bc_buffer.push_sequence(state_seq, action_seq)
    

    def plot_ab(self):
        """
        Draw two series A and B with automatically derived step axes for any length of data.
        If one of the series is empty, only plot the other.
        """
        # Y-axis data
        a = self.a_values
        b = self.b_values
    
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot series A if it has data
        if a:
            steps_a = list(range(len(a)))
            ax.plot(steps_a, a, label='A', linestyle='-', linewidth=2)
    
        # Plot series B if it has data
        if b:
            steps_b = list(range(len(b)))
            ax.plot(steps_b, b, label='B', linestyle='--', linewidth=2)
    
        # If both are empty, show a warning text
        if not a and not b:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
    
        # Set labels and title
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Series A vs B', fontsize=14)
    
        # Legend and grid (legend only if at least one series plotted)
        if a or b:
            ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
        # Tight layout and display
        fig.tight_layout()
        plt.show()
    
    

    
        
    def reset_world(self):
        """重置整个 Webots 世界"""
        print("Resetting world...")
        self.worldReload()  # 使用 Supervisor 的 worldReload 方法
        self.step(self.timeStep)  # 让 Webots 重新加载世界

    def get_translation(self):
        return self.translation.getSFVec3f()
        
    def get_yaw(self):
        """获取 NAO 机器人绕 Z 轴的旋转角度（偏航角），直接通过 IMU 获取"""
        roll, pitch, yaw = self.imu.getRollPitchYaw()  # 获取 IMU 的 roll, pitch, yaw
        return yaw  # 返回 yaw 角（单位：度）
        
    def get_plane_movement(self, yaw_threshold=5.0):
        """
        计算 NAO 机器人在平面上的运动情况，并返回：
          - 'distance_moved'：本步沿世界 X+ 方向的净前进量
          - 'total_distance'：从起始位置沿 X 轴的累计净前进量
          - 'total_angle'：当前相对于 initYaw 的偏航角
          - 'angle_reward'：基于偏航变化的奖励
        """
        # 1) 读取当前位置与偏航
        self.current_position = self.get_translation()  # [x, y, z]
        self.current_yaw      = self.get_yaw()
    
        # 2) 本步平移向量
        dx = self.current_position[0] - self.pre_position[0]
        dy = self.current_position[1] - self.pre_position[1]
    
        # 3) 投影到世界 X+ 轴上的前进量 (cos 0 = 1, sin 0 = 0)
        distance_moved = dx # 简化为 dx
        

        # 4) 累计净前进量（从 start_position 投影到 X 轴）
        total_distance = self.current_position[0] - self.start_position[0]
    
        # 5) 计算偏航角奖励
        first_angle  = self.start_yaw - self.initYaw
        second_angle = self.current_yaw - self.initYaw
        total_angle  = second_angle
        angle_diff   = abs(abs(first_angle) - abs(second_angle))
        # 这里保留你原来的逻辑，也可以简化为 -k*|second_angle|
        if angle_diff < 0.001:
            angle_reward = 0.0
        else:
            # 偏离越大，奖励越低；收敛时奖励正值
            angle_reward = (abs(first_angle) - abs(second_angle)) * 100
    
        # 6) 更新 pre_position
        self.pre_position = list(self.current_position)
    
        return {
            'distance_moved': distance_moved,
            'total_distance': total_distance,
            'total_angle':    total_angle,
            'angle_reward':   angle_reward
        }
    
    
    def get_line_movement():
        return 

    def apply_action_to_robot(self, action):
        """将一个动作向量映射并应用到机器人各个关节"""
        joint_order = [
            'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        for i, name in enumerate(joint_order):
            self.motors[name].setPosition(action[i].item())
    


    def get_center_of_mass(self):
     
        return self.get_translation()[2]
    def get_init_position(self):
        RHipRoll_position = self.RHipRollSensor.getValue()
        LHipRoll_position = self.LHipRollSensor.getValue()
        RHipYawPitch_position = self.RHipYawPitchSensor.getValue()
        LHipYawPitch_position = self.LHipYawPitchSensor.getValue()
        RHipPitch_position = self.RHipPitchSensor.getValue()
        LHipPitch_position = self.LHipPitchSensor.getValue()
        RKneePitch_position = self.RKneePitchSensor.getValue()
        LKneePitch_position = self.LKneePitchSensor.getValue()
      
        RAnklePitch_position = self.RAnklePitchSensor.getValue()
        LAnklePitch_position = self.LAnklePitchSensor.getValue()
        RAnkleRoll_position = self.RAnkleRollSensor.getValue()
        LAnkleRoll_position = self.LAnkleRollSensor.getValue()
        init_postion=self.get_translation()
        return([RHipRoll_position, LHipRoll_position,RHipYawPitch_position,LHipYawPitch_position,RHipPitch_position,LHipPitch_position, RKneePitch_position,LKneePitch_position,
                RAnklePitch_position,LAnklePitch_position,RAnkleRoll_position , LAnkleRoll_position,init_postion])
        

    def get_current_state(self, step):
        # === 1. 姿态信息 ===
        roll, pitch, _ = self.imu.getRollPitchYaw()
        # 读取世界坐标系下的绝对朝向（axis–angle → yaw）
        x, y, z, theta = self.rotation.getSFRotation()
        qw = math.cos(theta / 2)
        sin_t2 = math.sin(theta / 2)
        qx = x * sin_t2;  qy = y * sin_t2;  qz = z * sin_t2
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        gyro_x, gyro_y, _ = self.gyro.getValues()

        # === 2. 步态相位 ===
        phase = (step % SEQUENCE_LENGTH) / SEQUENCE_LENGTH

        # === 3. 关节角度 ===
        joint_order = [
            'RHipYawPitch','LHipYawPitch',
            'RHipRoll',    'LHipRoll',
            'RHipPitch',   'LHipPitch',
            'RKneePitch',  'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll',  'LAnkleRoll'
        ]
        joint_angles = [self.sensors[name+'S'].getValue() for name in joint_order]

        # === 4. 质心高度 cm 归一化 ===
        cm_raw = self.get_translation()[2]   # 例如 0.2–0.5 米
        CM_MIN, CM_MAX = 0.6, 0.72            # 根据实际机器人高度范围调整
        cm_norm = (cm_raw - CM_MIN) / (CM_MAX - CM_MIN)
        cm_norm = min(max(cm_norm, 0.0), 1.0)  # 截断到 [0,1]

        # === 5. 拼接状态向量 ===
        state = [
            phase,
            roll,
            yaw,
            gyro_x,
            gyro_y,
            cm_norm
        ] + joint_angles
        #print(state)
        # === 6. 记录日志 ===
        # 将当前 state 存入 state_logs 用于后续统一绘图
        self.state_logs.append(state)

        return state
        
      



    def calculate_reward(self, next_state, prev_state, action, prev_action,
                          angle_reward, dm,
                         break_flag, trunc, step_idx,base_reward):
        reward = 0

  
        
        phase = next_state[0]  
        step_ratio = (step_idx + 1) / trunc  
        roll,  yaw = next_state[1:3]  
        self.a_values.append(yaw)   
        gyro = next_state[3:5]  
        prev_roll, prev_yaw = prev_state[1:3]  

        

        yaw_reward=abs(yaw/10)
        self.b_values.append(yaw_reward)
        reward-=yaw_reward
        sd=dm

        move_reward=dm*10
        move_reward=max(min(0.2,move_reward),-0.2)
 
        reward+=move_reward
        
  

        if break_flag:
            if step_ratio < 0.8:
                done_penalty = -0.1
            else:
                a = 10
                max_growth = math.exp(a * 0.2) - 1
                growth = math.exp(a * (step_ratio - 0.8)) - 1
                scaled = growth / max_growth
                done_penalty = -0.01 - (4.99 * scaled)
  
        else:
            done_penalty = 0.0
        reward += done_penalty
        
        


        smoothness_threshold = 0.2  # 设置平滑性奖励的阈值
        action_diff = torch.sum((action - prev_action) ** 2).item()
        #print("action_diff",action_diff)
        if step_idx ==0:
            if action_diff >3.5:
                action_diff=0.2
            else:
                action_diff=0         
        elif action_diff > smoothness_threshold  :
            action_diff= action_diff   # 惩罚动作变化过大
        else:
            action_diff=0
        reward -= action_diff 
        
        """
        # —— 在函数前面定义所有阈值和权重 —— 
        yaw_small_thresh    = 0.2    # 小偏航阈值（rad）
        diff_thresh         = 0.01   # 偏航变化量阈值（rad）
        reduce_weight_small = 1.0    # 小偏航修正奖励权重
        reduce_weight_large = 2.5    # 大偏航修正奖励权重
        large_weight        = 1.5    # 大偏航惩罚权重
        
        # —— 计算当前偏航及差值 —— 
        yaw_amt      = abs(yaw)
        yaw_diff_amt = yaw_amt - abs(prev_yaw)
        #print("yaw_amt ,prev_yaw,yaw_diff_amt",yaw_amt,abs(prev_yaw),yaw_diff_amt  )
        
        # —— 单独计算 yaw_reward —— 
        yaw_reward = 0.0
        if yaw_diff_amt < 0:
            # A. 偏航在减少 → 纠正奖励
            w = reduce_weight_small if yaw_amt <= yaw_small_thresh else reduce_weight_large
            yaw_reward = - w * yaw_diff_amt
        
        else:
            if yaw_amt < yaw_small_thresh and abs(yaw_diff_amt) < diff_thresh:
                # B1. 小偏航且变化小 → 不惩罚
                yaw_reward = 0.0
            elif yaw_amt > yaw_small_thresh:
                # B2. 偏航量大 → 线性大惩罚
                yaw_excess = yaw_amt - yaw_small_thresh
                yaw_reward = - large_weight * yaw_excess
            else:
                # B3. 小偏航但变化大 → 惩罚 = 变化量
                yaw_reward = - abs(yaw_diff_amt)
          """
        """  
        yaw_reward = - abs(yaw_diff_amt)
        heading_factor = torch.cos(yaw_reward)

        yaw_reward=max(min( yaw_reward, 0.2),-0.2)
        
        # —— 把 yaw_reward 加到总 reward 上 —— 
        reward += yaw_reward        
        """
        #reward += base_reward*2
    
        # === 9. 限制总奖励范围 ===
        reward = max(min(reward, 3), -3)
        if reward ==3 or reward == -3:
            print("extrem value")
        #self.a_values.append(reward)
        self.a_values.append(reward)
     
        
        print(f"[Reward Breakdown] total: {reward:.3f} | "
              f"yaw_reward: {yaw_reward:.3f} | "
              f"done_penalty: {done_penalty:.3f} | "
              f"move_reward: {move_reward:.3f} | "
              
              f" action_diff:{ -action_diff:.3f}|"
              )
        
        
        return reward


    

    
    def resetRobot(self, INITIAL_STATE):

        self.step(self.timeStep)

        joint_order = [
            'RHipRoll', 'LHipRoll',
            'RHipYawPitch', 'LHipYawPitch',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
    
        for joint_name, angle in zip(joint_order, INITIAL_STATE[:12]):
            if joint_name in self.motors:
                self.motors[joint_name].setPosition(angle)
            else:
                print(f" Motor '{joint_name}' not found!")
    
        
        self.translation.setSFVec3f(INITIAL_STATE[12])  
        self.rotation.setSFRotation([0, 0, 1, 0])        
    
        
        for _ in range(64):
            self.step(self.timeStep)
    

                


    def train_model(self,num_episodes,use_noise=True,noise_std=0.0,train_interval=5,max_falls=20,freeze_critic_steps=0):
        fall_counter = 0
        episodes_since_train = 0
        self.resetRobot(INITIAL_STATE)
        self.step(self.timeStep)
        self.pre_position=self.get_translation()
        step_count = 0
        prev_action = torch.zeros_like(torch.tensor(INITIAL_ACTIONS[0], dtype=torch.float32, device=self.ddpg.device))
        best_q_value = -float("inf")
        best_model_path = "best_ddpg_model.pth"
        a_loss, c_loss = None, None  
        if freeze_critic_steps > 0:
            self.ddpg.freeze_critic()
            print(f"Critic frozen for first {freeze_critic_steps} training updates.")
        best_reward = -float("inf")
        recent_rewards = deque(maxlen=3)  
        
        for ep in range(num_episodes):
            print(f"\n Episode {ep} (Falls so far: {fall_counter})")

            
            
            self.step(self.timeStep)
            self.start_position = self.get_translation()
    
           
            raw = self.get_current_state(0)
            state = torch.tensor(raw, dtype=torch.float32, device=self.ddpg.device)
            episode_rewards = []
    
            seq_states      = []
            seq_actions     = []
            seq_next_states = []
            seq_angle_r     = []
            seq_dm          = []
            seq_preva       = []
            fell            = False
    
            
            for t in range(SEQUENCE_LENGTH):
                step_count += 1
                self.steps.append(step_count)
    
                
                if use_noise:
                    base   = torch.tensor(
                        INITIAL_ACTIONS[t % len(INITIAL_ACTIONS)],
                        dtype=torch.float32, device=self.ddpg.device
                    )
                    action = base + torch.normal(0, noise_std, size=base.shape, device=self.ddpg.device)
                else:
                    self.ddpg.actor.eval()
                    with torch.no_grad():
                        action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)
    
               
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                raw_next = self.get_current_state(t + 1)
                next_state = torch.tensor(raw_next, dtype=torch.float32, device=self.ddpg.device)
    
               
                if self.get_center_of_mass() < 0.6:
                    print(" Falling")
                    fall_counter += 1
                    fell = True
               
                
                pm     = self.get_plane_movement()
                angle_r= pm["angle_reward"]
                dm     = pm["distance_moved"]
               
    
                
                seq_states.append(state.cpu())
                seq_actions.append(action.detach().cpu())
                seq_next_states.append(next_state.cpu())
                seq_angle_r.append(angle_r)
                seq_dm.append(dm)
                #self.a_values.append(dm) 
                seq_preva.append(prev_action)
               
                state        = next_state
                prev_action  = action.detach().clone()

                if fell:
                    
                    self.resetRobot(INITIAL_STATE)
                    self.step(self.timeStep)
                    self.pre_position=self.get_translation()
                    prev_action = torch.zeros_like(torch.tensor(INITIAL_ACTIONS[0], dtype=torch.float32, device=self.ddpg.device))
                    break_flag = True
                    break
                
    
            base_reward = pm["total_distance"]
            seq_len     = len(seq_states)
            
            if fell:
      
                seq_dones = [False] * (seq_len - 1) + [True]
            else:
                seq_dones = [False] * seq_len
            
            for i in range(seq_len):
                
                s      = seq_states[i].clone().detach().to(self.ddpg.device)
                a      = seq_actions[i].clone().detach().to(self.ddpg.device)
                next_s = seq_next_states[i].clone().detach().to(self.ddpg.device)
                pa     = seq_preva[i].clone().detach().to(self.ddpg.device)
   
                
                r = self.calculate_reward(
                    next_s,
                    s,
                    a,
                    pa,
                    seq_angle_r[i],
                    seq_dm[i],
                    fell,
                    seq_len,
                    i,
                    base_reward
                )
                episode_rewards.append(r) 
                """
                if use_noise:
                    self.a_values.append(r)    
                else:
                    self.b_values.append(r)    
                """
                
                # 推入 replay buffer
                self.ddpg.replay_buffer.push(
                    s,
                    a,
                    r,
                    next_s,
                    seq_dones[i]
                )
    
            print(f" Pushed sequence of length {seq_len} (fell={fell}) to buffer")

            if fell:
                print(" Immediate  update after fall")
                if len(self.ddpg.replay_buffer) > self.ddpg.batch_size:
                    a_loss, c_loss = self.ddpg.train()
                    episodes_since_train = 0
            else:
                episodes_since_train += 1
                if episodes_since_train >= train_interval:
                    
                    a_loss, c_loss = self.ddpg.train()
                    episodes_since_train = 0                    
                    
                else:
                    a_loss, c_loss = None, None

            if freeze_critic_steps > 0:
                freeze_critic_steps -= 1
                print(f" Critic freeze steps remaining: {freeze_critic_steps}")
                if freeze_critic_steps == 0:
                    self.ddpg.unfreeze_critic()
                    print(" Critic unfrozen — resume normal training.")

            if a_loss is not None:
                print(f"Train ⇒ Actor loss={a_loss:.6f}, Critic loss={c_loss:.6f}")
                self.ddpg.replay_buffer.clear()
                #print(" Replay buffer cleared")

                
            if fall_counter >= max_falls:
                print(f"Too many falls ({fall_counter}), stopping early.")
                break
                
                
            total_reward = sum(episode_rewards)  
            recent_rewards.append(total_reward)
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
            #self.a_values.append(yaw) 
            #self.b_values.append(total_reward) 
            print(f"Episode {ep} reward: {total_reward:.2f}, Smoothed: {avg_recent_reward:.2f}, Best so far: {best_reward:.2f}")
            
            if avg_recent_reward > best_reward:
                best_reward = avg_recent_reward
                self.ddpg.save_checkpoint(filepath=best_model_path)
                print(f"Best model saved with smoothed reward={best_reward:.2f} ➜ {best_model_path}")

       
        self.ddpg._plot_training_summary(save_path="training_summary")
    
    
    def run_trained_actor(self,
                          checkpoint_path: str = "ddpg_checkpoint.pth",
                          num_episodes: int = 10,
                          stop_x: float = None,
                          max_steps: int = 1000):

        print(f"Loading checkpoint from {checkpoint_path} …")
        self.ddpg.load_checkpoint(filepath=checkpoint_path)
        self.ddpg.actor.eval()
    
        results = []
        dt = self.timeStep / 1000.0  
    
        for ep in range(num_episodes):
            print(f"\n [Eval] Episode {ep + 1}/{num_episodes}")
            self.resetRobot(INITIAL_STATE)
            self.step(self.timeStep)
            start_pos = self.get_translation()
            start_x = start_pos[0]
    
            raw = self.get_current_state(0)
            state = torch.tensor(raw, dtype=torch.float32, device=self.ddpg.device)
    
            fell = False
            reach_time = None
    
            for t in range(max_steps):
                with torch.no_grad():
                    action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)
    
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
    
                pos = self.get_translation()
                cur_x = pos[0]
                cur_y = pos[1]
    
                if stop_x is not None and cur_x >= stop_x:
                    reach_time = (t + 1) * dt
                    print(f"Reached x ≥ {stop_x:.3f} m at t={reach_time:.2f} s")
                    break
    
                raw_next = self.get_current_state(t + 1)
                state = torch.tensor(raw_next, dtype=torch.float32, device=self.ddpg.device)
    
                if self.get_center_of_mass() < 0.6:
                    print("  Fell at step", t + 1)
                    fell = True
                    break
    
            end_pos = self.get_translation()
            forward_dist = end_pos[0] - start_x
            lateral_dist = end_pos[1] - start_pos[1]
            simulated_steps = t + 1
            elapsed_time = simulated_steps * dt
    
            print(f"→ Episode {ep + 1}: forward={forward_dist:.3f} m, "
                  f"lateral={lateral_dist:.3f} m, fell={fell}, "
                  f"time={elapsed_time:.2f} s, reach_time={reach_time}")
    
            results.append({
                "episode": ep + 1,
                "forward": forward_dist,
                "lateral": lateral_dist,
                "fell": fell,
                "time": elapsed_time,         
                "steps": simulated_steps,     
                "reach_time": reach_time      
            })
    
        return results

    def run_walk_sequence_until_x(self,
                                   target_x: float = 4.5,
                                   gait_sequence = INITIAL_ACTIONS,
                                   max_loops: int = 200):

        print(f" Running fixed gait until reaching x = {target_x} m …")
        dt = self.timeStep / 1000.0  
    
        self.resetRobot(INITIAL_STATE)
        self.step(self.timeStep)
        start_pos = self.get_translation()
        start_x = start_pos[0]
    
        fell = False
        step_counter = 0
    
        for loop in range(max_loops):
            for t in range(len(gait_sequence)):
                action = torch.tensor(gait_sequence[t], dtype=torch.float32, device=self.ddpg.device)
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                step_counter += 1
    
                pos = self.get_translation()
                cur_x = pos[0]
    
                if self.get_center_of_mass() < 0.6:
                    print(f" Fell at step {step_counter}")
                    fell = True
                    break
    
                if cur_x >= target_x:
                    elapsed_time = step_counter * dt
                    print(f" Reached x = {target_x:.2f} m in {elapsed_time:.2f} s over {step_counter} steps")
                    return {
                        "reached": True,
                        "time": elapsed_time,
                        "steps": step_counter,
                        "forward": cur_x - start_x,
                        "fell": False
                    }
    
            if fell:
                break
    

        print(f" Did not reach target x = {target_x:.2f} m, final x = {cur_x:.2f} m")
        return {
            "reached": False,
            "time": step_counter * dt,
            "steps": step_counter,
            "forward": cur_x - start_x,
            "fell": True
        }
        
    def plot_state_trajectories(self, plots_per_fig: int = 6):

        arr = np.array(self.state_logs, dtype=float)  # shape = (total_steps, state_dim)
        total_steps, state_dim = arr.shape

        
        if total_steps % SEQUENCE_LENGTH == 0:
            num_eps = total_steps // SEQUENCE_LENGTH
            all_states = arr.reshape(num_eps, SEQUENCE_LENGTH, state_dim)
            seq_len = SEQUENCE_LENGTH
        else:
            all_states = arr[np.newaxis, ...]  # shape = (1, total_steps, state_dim)
            num_eps = 1
            seq_len = total_steps

        num_figs = math.ceil(state_dim / plots_per_fig)
        ncols = math.ceil(plots_per_fig / 2)
        nrows = 2

        for fig_idx in range(num_figs):
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     figsize=(4 * ncols, 4 * nrows))
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                dim_idx = fig_idx * plots_per_fig + i
                if dim_idx >= state_dim:
                    ax.axis('off')
                    continue

                name = self.columns[dim_idx]
                for ep in range(num_eps):
                    ax.plot(
                        range(seq_len),
                        all_states[ep, :, dim_idx],
                        alpha=0.6,
                        label=f"ep{ep+1}" if ep == 0 else None
                    )
                ax.set_title(name)
                ax.set_xlabel("Step")
                ax.set_ylabel(name)
                ax.grid(True)
                if num_eps <= 5:
                    ax.legend()

            plt.tight_layout()
            plt.show()
    




controller = Sprinter()  
controller.initialize()
ckpt_path = CKPTPATH
ckptl_path = CKPTLPATH

#controller.ddpg.load_bc()
#controller.run_bc_once()
#controller.train_model(TRAIN_EPISODES)
"""
controller.ddpg.load_checkpoint( ckpt_path)
#controller.ddpg.load_bc()
controller.train_model(NUM_EPISODES,use_noise=False) 
controller.ddpg.save_checkpoint(ckpt_path )
"""
#controller.train_model(NUM_EPISODES,use_noise=False,freeze_critic_steps=10) 
#controller.plot_state_trajectories()

results = controller.run_trained_actor(
    checkpoint_path="best_ddpg_model_rd.pth",
    num_episodes=1,
    stop_x=4.5,          
    max_steps=6000        
)
#controller.run_walk_sequence_until_x(target_x=4.5)
#checkpoint_path="ddpg_checkpoint.pth"
#checkpoint_path="best_ddpg_model.pth",

#controller.plot_ab()









