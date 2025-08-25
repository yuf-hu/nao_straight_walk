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
import os,csv,json
import copy
from pathlib import Path
from datetime import datetime
from helperfunction import (
    load_initial_actions_csv,
    _prepare_logging,
    _append_metric,
    _save_meta,
    save_checkpoint,
    load_checkpoint,
    plot_q_reward_over_updates_gs,
    plot_q_reward_from_csv_gs,
    rotate_metrics_csv,
    load_bc
)
STATE_DIM = 18
ACTION_DIM = 100
TRAIN_EPISODES=10
NUM_EPISODES =100
NUM_BC_EPISODES = 100
SEQUENCE_LENGTH=35
INITIAL_STATE=[2.813063620922607e-07, -2.8354001787624697e-07, 2.661361167672216e-07, 2.680192760864757e-07, 2.967163694572752e-07, 2.9883194662671264e-07, 2.4820736847423014e-07, 2.4949180207911914e-07, 1.9713857411518186e-07, 1.9756885276656546e-07, -1.866298555697199e-07, 1.8715301552270632e-07, [-5.021522061737612,  0.9999547666320551, 0.7058214811932133]]
ACTOR_LR=1e-4
CRITIC_LR=1e-4
MAX_FALLS= 20
CKPTPATH="ddpg_checkpoint.pth"
CKPTLPATH="ddpg_checkpoint_false.pth"
 
SMOOTH_K=5 
INITIAL_ACTIONS = load_initial_actions_csv(
    "initial_actions.csv",   # 同目录下的文件名
    expected_dim=10,         # 动作维度
    to_radians=False,        # 如果是度就设 True
    device="cpu",
    dtype=torch.float32
)

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
     
        self.last_train_metrics={}
        self._train_logs = {
            "rewards": [], "target_qs": [], "current_qs": [],
            "actor_losses": [], "critic_losses": [], "q_vs_r_pairs": [] 
        }
        self.actor_update_start = 0

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * param.data)
    


    def train(self, critic_updates: int = 2):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
    
        states, actions, rewards, next_states, dones, _ = \
            self.replay_buffer.sample_batch(self.batch_size)
    
        device = self.device
        states      = states.to(device)
        actions     = actions.to(device)
        rewards     = rewards.to(device).view(-1)     # 扁平化便于统计
        next_states = next_states.to(device)
        dones       = dones.to(device).view(-1)
    
        # --- 无梯度统计：Q/Reward 均值等 + 最小/最大（用于波动范围） ---
        with torch.no_grad():
            next_a   = self.actor_target(next_states)
            next_q   = self.critic_target(next_states, next_a).view(-1)
            target_q = rewards + self.gamma * (1 - dones) * next_q
    
            current_q = self.critic(states, actions).view(-1)
            td_mse    = F.mse_loss(current_q, target_q).item()
            policy_q  = self.critic(states, self.actor(states)).mean().item()
    
            m = {
                "replay_size": len(self.replay_buffer),
                "batch_size": int(self.batch_size),
                "critic_updates": int(critic_updates),
    
                # —— Q/Reward 的均值/标准差/最小/最大 —— #
                "current_q_mean": current_q.mean().item(),
                "current_q_std":  current_q.std(unbiased=False).item(),
                "current_q_min":  current_q.min().item(),
                "current_q_max":  current_q.max().item(),
    
                "target_q_mean":  target_q.mean().item(),
    
                "reward_mean":    rewards.mean().item(),
                "reward_std":     rewards.std(unbiased=False).item(),
                "reward_min":     rewards.min().item(),
                "reward_max":     rewards.max().item(),
    
                "policy_q_mean":  policy_q,
                "td_mse":         td_mse,
                "done_ratio":     dones.float().mean().item(),
            }
    
        # --- 工具：全局梯度范数 ---
        def _grad_global_norm(module):
            tot = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    tot += p.grad.data.pow(2).sum().item()
            return tot ** 0.5
    
        avg_critic_loss = 0.0
        critic_grad_norms = []
    
        # --- Critic 多次更新 ---
        if any(p.requires_grad for p in self.critic.parameters()):
            total_critic_loss = 0.0
            for _ in range(critic_updates):
                with torch.no_grad():
                    next_a   = self.actor_target(next_states)
                    next_q   = self.critic_target(next_states, next_a).view(-1)
                    target_q = rewards + self.gamma * (1 - dones) * next_q
    
                current_q   = self.critic(states, actions).view(-1)
                critic_loss = F.mse_loss(current_q, target_q)
    
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norms.append(_grad_global_norm(self.critic))
                self.critic_optimizer.step()
                self.soft_update(self.critic, self.critic_target)
                total_critic_loss += critic_loss.item()
    
            avg_critic_loss = total_critic_loss / critic_updates
    
        # --- Actor 更新 ---
        pred_a     = self.actor(states)
        actor_loss = - self.critic(states, pred_a).mean()
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = _grad_global_norm(self.actor)
        self.actor_optimizer.step()
        self.soft_update(self.actor, self.actor_target)
    
        # --- 缓存给外部（train_model 会读这些键） ---
        m.update({
            "actor_loss": actor_loss.item(),
            "critic_loss": avg_critic_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm_mean": (sum(critic_grad_norms) / max(1, len(critic_grad_norms))) if critic_grad_norms else 0.0,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
        })
        self.last_train_metrics = m
    
        # —— 每次 train() 完成后：打印当前批次 Q/Reward “分布摘要” —— #
        print(
            "[train] "
            f"Qμ={m['current_q_mean']:.3f} σ={m['current_q_std']:.3f} "
            f"range=[{m['current_q_min']:.3f},{m['current_q_max']:.3f}]  |  "
            f"Rμ={m['reward_mean']:.3f} σ={m['reward_std']:.3f} "
            f"range=[{m['reward_min']:.3f},{m['reward_max']:.3f}]  "
            f"TDmse={m['td_mse']:.4f}"
        )
    
        return actor_loss.item(), avg_critic_loss



    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state)
        return action.squeeze(0)
               
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

        self.columns = [
            "phase", "roll", "yaw", "gyro_x", "gyro_y", "cm_norm",
            "RHipYawPitch","LHipYawPitch","RHipRoll","LHipRoll",
            "RHipPitch","LHipPitch","RKneePitch","LKneePitch",
            "RAnklePitch","LAnklePitch","RAnkleRoll","LAnkleRoll"
        ]


    
    def print_joint_ranges(self):
        for name in self.joint_names:
            motor = self.getDevice(name)
            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            print(f"{name}: Min = {min_pos:.3f}, Max = {max_pos:.3f}")
  

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


        return state
        
      



    def calculate_reward(self, next_state, prev_state, action, prev_action,
                          angle_reward, dm,
                         break_flag, trunc, step_idx,base_reward):
        reward = 0

  
        
        phase = next_state[0]  
        step_ratio = (step_idx + 1) / trunc  
        roll,  yaw = next_state[1:3]  
        
        gyro = next_state[3:5]  
        prev_roll, prev_yaw = prev_state[1:3]  

        

        yaw_reward=abs(yaw/10)
        
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
        
    
        # === 9. 限制总奖励范围 ===
        reward = max(min(reward, 3), -3)
        if reward ==3 or reward == -3:
            print("extrem value")
        #self.a_values.append(reward)
       
     
        """
        print(f"[Reward Breakdown] total: {reward:.3f} | "
              f"yaw_reward: {yaw_reward:.3f} | "
              f"done_penalty: {done_penalty:.3f} | "
              f"move_reward: {move_reward:.3f} | "
              
              f" action_diff:{ -action_diff:.3f}|"
              )
        
        """
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
    

                




    def train_model(self,
                    num_episodes: int,
                    train_interval: int = 5,
                    best_model_path: str = "check_points/best_ddpg_model.pth",
                    *,
                    log_dir: str = "runs/walk_exp1",
                    resume_logs: bool = True,
                    save_meta_every: int = 5):        # 每多少次“更新”保存一次 meta
        """
        精简训练：
        - 仅在发生一次“参数更新”(self.ddpg.train()) 时：
            * 追加一行 CSV (type='update')
            * 打印一行到控制台
        - best 模型仍按平滑奖励判定与保存（但不打印）
        - global_step 以“更新步”为单位累加
        """
        # 1) 日志初始化/衔接（会恢复 global_step / ep_offset / best_reward）
        if not resume_logs:
            rotate_metrics_csv(log_dir)
            self.global_step = 0
            self.ep_offset   = 0
            self.best_reward = float("-inf")
        _prepare_logging(self, log_dir=log_dir, resume=resume_logs)
        print(f"[resume-check] log_dir={self.log_dir}")
        print(f"[resume-check] meta -> global_step={self.global_step}  ep_offset={self.ep_offset}  best={self.best_reward:.3f}")

        # 2) 常规初始化
        fall_counter = 0
        episodes_since_train = 0
        updates_done = 0  # 本次调用内的更新次数

        self.resetRobot(INITIAL_STATE)
        self.step(self.timeStep)
        self.pre_position = self.get_translation()

        prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
        a_loss, c_loss = None, None

        recent_rewards = deque(maxlen=SMOOTH_K)
        best_reward = float(getattr(self, "best_reward", float("-inf")))

        episodes_done = 0
        Qmu_list, Rmu_list, Qrng_list = [], [], []
        # 3) 主循环
        for ep in range(num_episodes):
            self.step(self.timeStep)
            self.start_position = self.get_translation()

            raw = self.get_current_state(0)
            state = torch.tensor(raw, dtype=torch.float32, device=self.ddpg.device)

            episode_rewards = []

            seq_states, seq_actions, seq_next_states = [], [], []
            seq_angle_r, seq_dm, seq_preva = [], [], []
            fell = False
            pm = {"total_distance": 0.0, "angle_reward": 0.0, "distance_moved": 0.0}

            for t in range(SEQUENCE_LENGTH):
                self.ddpg.actor.eval()
                with torch.no_grad():
                    action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)

                self.apply_action_to_robot(action)
                self.step(self.timeStep)

                raw_next = self.get_current_state(t + 1)
                next_state = torch.tensor(raw_next, dtype=torch.float32, device=self.ddpg.device)

                if self.get_center_of_mass() < 0.6:
                    fall_counter += 1
                    fell = True

                pm = self.get_plane_movement()
                angle_r = pm["angle_reward"]
                dm = pm["distance_moved"]

                seq_states.append(state.cpu())
                seq_actions.append(action.detach().cpu())
                seq_next_states.append(next_state.cpu())
                seq_angle_r.append(angle_r)
                seq_dm.append(dm)
                seq_preva.append(prev_action)

                state = next_state
                prev_action = action.detach().clone()

                if fell:
                    self.resetRobot(INITIAL_STATE)
                    self.step(self.timeStep)
                    self.pre_position = self.get_translation()
                    prev_action = torch.zeros_like(INITIAL_ACTIONS[0])
                    break

            base_reward = pm["total_distance"]
            print("total_distance",base_reward)
            seq_len = len(seq_states)
            if fell:
                seq_dones = [False] * max(0, seq_len - 1) + [True] if seq_len > 0 else [True]
            else:
                seq_dones = [False] * seq_len

            for i in range(seq_len):
                s = seq_states[i].clone().detach().to(self.ddpg.device)
                a = seq_actions[i].clone().detach().to(self.ddpg.device)
                next_s = seq_next_states[i].clone().detach().to(self.ddpg.device)
                pa = seq_preva[i].clone().detach().to(self.ddpg.device)

                r = self.calculate_reward(
                    next_s, s, a, pa,
                    seq_angle_r[i], seq_dm[i],
                    fell, seq_len, i, base_reward
                )
                episode_rewards.append(r)
                self.ddpg.replay_buffer.push(s, a, r, next_s, seq_dones[i])

            # —— 尝试触发一次参数更新 —— #
            updated_this_ep = False
            buffer_before_clear = len(self.ddpg.replay_buffer)  # 打印用

            if fell:
                if len(self.ddpg.replay_buffer) > self.ddpg.batch_size:
                    a_loss, c_loss = self.ddpg.train()
                    episodes_since_train = 0
                    updated_this_ep = True
            else:
                episodes_since_train += 1
                if episodes_since_train >= train_interval and len(self.ddpg.replay_buffer) > self.ddpg.batch_size:
                    a_loss, c_loss = self.ddpg.train()
                    episodes_since_train = 0
                    updated_this_ep = True
                else:
                    a_loss, c_loss = None, None

            if a_loss is not None:
                self.ddpg.replay_buffer.clear()

            # —— 奖励与平滑（即便未更新也要计算，用于 best 判定）—— #
            total_reward = float(sum(episode_rewards)) if episode_rewards else 0.0
            recent_rewards.append(total_reward)
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
            # —— Best 判定并保存（按平滑回报）——
            cur_best = float(getattr(self, "best_reward", float("-inf")))
            if avg_recent_reward > cur_best + 1e-6:  # 小epsilon防抖
                self.best_reward = avg_recent_reward  # 更新到实例，便于 _save_meta 持久化
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                save_checkpoint(self.ddpg, filepath=best_model_path)
                # 可选：只在发生更新时打印；否则这里也可以打印一行
                print(f"[best] gs={self.global_step} ep={int(self.ep_offset)+ep} "
                      f"avgR={avg_recent_reward:.3f} -> saved {best_model_path}")

            if updated_this_ep:
                self.global_step = int(getattr(self, "global_step", 0)) + 1
                updates_done += 1
            
                m = getattr(self.ddpg, "last_train_metrics", {}) or {}
            
                # —— 取均值与波动范围 —— #
                qmu = m.get("current_q_mean")
                rmu = m.get("reward_mean")
                qmin = m.get("current_q_min")
                qmax = m.get("current_q_max")
                qrng = (qmax - qmin) if (qmin is not None and qmax is not None) else None
            
                # —— 记到 CSV 的 note —— #
                def fmt(x, spec=".3f"):
                    return "NA" if x is None else format(x, spec)
            
                _append_metric(self, {
                    "global_step": self.global_step,
                    "episode": int(self.ep_offset) + ep,
                    "type": "update",
                    "actor_loss": float(a_loss) if a_loss is not None else None,
                    "critic_loss": float(c_loss) if c_loss is not None else None,
                    "reward": total_reward,
                    "avg_recent_reward": avg_recent_reward,
                    "fell": bool(fell),
                    "seq_len": int(seq_len),
                    "note": (
                        f"Qμ={fmt(qmu)} Rμ={fmt(rmu)} Qrng={fmt(qrng)}; "
                        f"buffer={buffer_before_clear}"
                    )
                })
            
                # —— 追加到本轮统计 —— #
                if qmu is not None: Qmu_list.append(qmu)
                if rmu is not None: Rmu_list.append(rmu)
                if qrng is not None: Qrng_list.append(qrng)
                """
                # —— 控制台打印（只在更新时）——
                aL = f"{float(a_loss):.4f}" if a_loss is not None else "—"
                cL = f"{float(c_loss):.4f}" if c_loss is not None else "—"
                vt = f"{td_mse:.4f}" if td_mse is not None else "NA"
                vq = f"{pol_q:.4f}"  if pol_q  is not None else "NA"
                qa = f"{q_cur:.3f}/{q_tgt:.3f}" if (q_cur is not None and q_tgt is not None) else "NA"
                gA = f"{g_a:.2f}" if g_a is not None else "NA"
                gC = f"{g_c_mean:.2f}" if g_c_mean is not None else "NA"
                lrs= (f"{a_lr:.1e}/{c_lr:.1e}" if (a_lr is not None and c_lr is not None) else "NA")
                print(f"[upd {self.global_step}] ep={int(self.ep_offset)+ep}  "
                      f"a_loss={aL}  c_loss={cL}  td_mse={vt}  polQ={vq}  q={qa}  "
                      f"gradA={gA}  gradC={gC}  lrA/C={lrs}  "
                      f"avgR={avg_recent_reward:.3f}  R={total_reward:.3f}  "
                      f"len={seq_len}  fell={int(fell)}  buf={buffer_before_clear}")
                """

        episodes_done += 1

        # —— 收尾：推进 ep_offset（按“episode 数”），保存 meta —— #
        self.ep_offset = int(getattr(self, "ep_offset", 0)) + episodes_done
        _save_meta(self)
        n_upd = len(Qmu_list)
        if n_upd == 0:
            print("[summary] no parameter updates occurred; nothing to summarize.")
        else:
            # 保存 meta 后：
            plot_q_reward_over_updates_gs(self.log_dir, Qmu_list, Rmu_list, self.global_step)
            # 可选：再画一张“累计历史”的（跨多次重启，读 CSV）
            plot_q_reward_from_csv_gs(self.log_dir, self.global_step, lower_quantile=0.01)



    
    
    def run_trained_actor(self,
                          checkpoint_path: str = "ddpg_checkpoint.pth",
                          num_episodes: int = 10,
                          stop_x: float = None,
                          max_steps: int = 1000):
        print(f"Loading checkpoint from {checkpoint_path} …")
        load_checkpoint(host=self.ddpg,filepath=checkpoint_path)
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
            step_counter = 0

            for t in range(max_steps):
                with torch.no_grad():
                    action = self.ddpg.actor(state.unsqueeze(0)).squeeze(0)

                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                step_counter += 1

                pos = self.get_translation()
                cur_x = pos[0]
                cur_y = pos[1]

                if stop_x is not None and cur_x >= stop_x:
                    reach_time = step_counter * dt
                    print(f"Reached x ≥ {stop_x:.3f} m at t={reach_time:.2f} s")
                    break

                raw_next = self.get_current_state(step_counter)
                state = torch.tensor(raw_next, dtype=torch.float32, device=self.ddpg.device)

                if self.get_center_of_mass() < 0.6:
                    print("  Fell at step", step_counter)
                    fell = True
                    break

            end_pos = self.get_translation()
            forward_dist = end_pos[0] - start_x
            lateral_dist = end_pos[1] - start_pos[1]
            elapsed_time = step_counter * dt

            print(f"→ Episode {ep + 1}: forward={forward_dist:.3f} m, "
                  f"lateral={lateral_dist:.3f} m, fell={fell}, "
                  f"time={elapsed_time:.2f} s, steps={step_counter}, reach_time={reach_time}")

            results.append({
                "episode": ep + 1,
                "forward": forward_dist,
                "lateral": lateral_dist,
                "fell": fell,
                "time": elapsed_time,
                "steps": step_counter,
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
                action = gait_sequence[t]  # 已经是 tensor
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
        

    




controller = Sprinter()  
controller.initialize()
ckpt_path = CKPTPATH
ckptl_path = CKPTLPATH

load_checkpoint( host=controller.ddpg)
#load_bc(host=controller.ddpg)
#controller.run_bc_once()
controller.train_model(TRAIN_EPISODES,resume_logs=True)
save_checkpoint( host=controller.ddpg)
"""
load_checkpoint( ckpt_path)
#load_bc(host=self.ddpg)
controller.train_model(NUM_EPISODES,use_noise=False) 
save_checkpoint(host=self.ddpg,ckpt_path )
"""
#controller.train_model(NUM_EPISODES,use_noise=False,freeze_critic_steps=10) 

"""
results = controller.run_trained_actor(
    checkpoint_path="best_ddpg_model_rd.pth",
    num_episodes=1,
    stop_x=4.5,          
    max_steps=6000        
)
"""
#controller.run_walk_sequence_until_x(target_x=4.5)
#checkpoint_path="ddpg_checkpoint.pth"
#checkpoint_path="best_ddpg_model.pth",

#controller.plot_ab()









