from controller import Supervisor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from collections import deque
from helperfunction import load_initial_actions_csv

STATE_DIM = 18
ACTION_DIM = 10          # 你的 Actor 实际用了 10 维，请保持一致
SEQUENCE_LENGTH = 35
INITIAL_STATE=[2.813063620922607e-07, -2.8354001787624697e-07, 2.661361167672216e-07, 2.680192760864757e-07, 2.967163694572752e-07, 2.9883194662671264e-07, 2.4820736847423014e-07, 2.4949180207911914e-07, 1.9713857411518186e-07, 1.9756885276656546e-07, -1.866298555697199e-07, 1.8715301552270632e-07, [-5.021522061737612,  0.9999547666320551, 0.7058214811932133]]

# 读取固定步态（CSV → tensor 列表）
INITIAL_ACTIONS = load_initial_actions_csv(
    "initial_actions.csv", expected_dim=ACTION_DIM, device="cpu"
)

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
        self.register_buffer("joint_min", torch.tensor(
            [-0.379,-1.774,-0.092,-1.189,-0.397,-0.738,-1.774,-0.092,-1.186,-0.769], dtype=torch.float32))
        self.register_buffer("joint_max", torch.tensor(
            [ 0.790, 0.484, 2.113, 0.923, 0.769, 0.450, 0.484, 2.113, 0.932, 0.380], dtype=torch.float32))

    def forward(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.net[0].weight.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        raw = self.net(state)
        return torch.max(torch.min(raw, self.joint_max), self.joint_min)

class SequenceBCBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    def push_sequence(self, states, actions):
        s = [ (x.detach().cpu().tolist() if torch.is_tensor(x) else x) for x in states ]
        a = [ (x.detach().cpu().tolist() if torch.is_tensor(x) else x) for x in actions ]
        self.buffer.append((s, a))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    def flatten_sample(self, batch_size):
        import random
        data = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        all_s, all_a = [], []
        for s_seq, a_seq in data:
            all_s.extend(s_seq); all_a.extend(a_seq)
        return (torch.tensor(all_s, dtype=torch.float32),
                torch.tensor(all_a, dtype=torch.float32))
    def __len__(self): return len(self.buffer)
    def clear(self): self.buffer.clear()

class Sprinter(Supervisor):
    def __init__(self, auto_load_bc=True, bc_path="actor_bc.pth"):
        super().__init__()
        self.timeStep = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(STATE_DIM, 256, ACTION_DIM).to(self.device)
        self.bc_buffer = SequenceBCBuffer(capacity=10000)
        self.bc_losses = []  # 用于记录并在训练中打印

        # 可选：自动加载历史 BC
        if auto_load_bc and os.path.isfile(bc_path):
            try:
                self.load_bc_actor(bc_path)
                print(f"[BC] Loaded previous BC actor from: {bc_path}")
            except Exception as e:
                print(f"[BC] Failed to load BC actor from {bc_path}: {e}")

    def load_bc_actor(self, path="actor_bc.pth", strict=True, eval_mode=False):
        """读取之前保存的 BC Actor 权重。"""
        state_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state_dict, strict=strict)
        if eval_mode:
            self.actor.eval()
        else:
            self.actor.train()

    def save_bc_actor(self, path="actor_bc.pth"):
        torch.save(self.actor.state_dict(), path)
        print(f"[BC] Saved BC actor to: {path}")

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
        self.initYaw = 0
        print("self.initYaw", self.initYaw)
        self.pre_position = self.get_translation()
        self.start_position = self.get_translation()
        self.current_position = self.get_translation()
        self.pre_yaw = self.get_yaw()
        self.start_yaw = self.get_yaw()
        self.current_yaw = self.get_yaw()
        self.steps = []

        self.columns = [
            "phase", "roll", "yaw", "gyro_x", "gyro_y", "cm_norm",
            "RHipYawPitch","LHipYawPitch","RHipRoll","LHipRoll",
            "RHipPitch","LHipPitch","RKneePitch","LKneePitch",
            "RAnklePitch","LAnklePitch","RAnkleRoll","LAnkleRoll"
        ]

    def get_translation(self):
        return self.translation.getSFVec3f()

    def get_yaw(self):
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        return yaw

    def get_current_state(self, step):
        roll, pitch, _ = self.imu.getRollPitchYaw()
        x, y, z, theta = self.rotation.getSFRotation()
        qw = math.cos(theta / 2); s = math.sin(theta / 2)
        qx = x*s; qy = y*s; qz = z*s
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        gx, gy, _ = self.gyro.getValues()
        phase = (step % SEQUENCE_LENGTH) / SEQUENCE_LENGTH
        joint_order = ['RHipYawPitch','LHipYawPitch','RHipRoll','LHipRoll',
                       'RHipPitch','LHipPitch','RKneePitch','LKneePitch',
                       'RAnklePitch','LAnklePitch','RAnkleRoll','LAnkleRoll']
        joint_angles = [self.sensors[name+'S'].getValue() for name in joint_order]
        cm_raw = self.get_translation()[2]
        CM_MIN, CM_MAX = 0.6, 0.72
        cm_norm = (cm_raw - CM_MIN) / (CM_MAX - CM_MIN)
        cm_norm = min(max(cm_norm, 0.0), 1.0)
        return [phase, roll, yaw, gx, gy, cm_norm] + joint_angles

    def apply_action_to_robot(self, action):
        joint_order = ['LHipRoll','LHipPitch','LKneePitch','LAnklePitch','LAnkleRoll',
                       'RHipRoll','RHipPitch','RKneePitch','RAnklePitch','RAnkleRoll']
        for i, name in enumerate(joint_order):
            self.motors[name].setPosition(action[i].item())

    def get_plane_movement(self):
        cur = self.get_translation()
        dx = cur[0] - self.pre_position[0]
        self.pre_position = list(cur)
        return {"distance_moved": dx, "total_distance": cur[0] - self.start_position[0],
                "total_angle": self.get_yaw(), "angle_reward": 0.0}

    def collect_bc_data(self, num_episodes=40):
        self.bc_buffer.clear()
        for ep in range(num_episodes):
            self.pre_position = self.get_translation()
            state_seq, action_seq = [], []
            cur_state = self.get_current_state(0)
            for s in range(SEQUENCE_LENGTH):
                action = INITIAL_ACTIONS[s % len(INITIAL_ACTIONS)]
                self.apply_action_to_robot(action)
                self.step(self.timeStep)
                next_state = self.get_current_state(s)
                state_seq.append(cur_state)
                action_seq.append(action)
                pm = self.get_plane_movement()
                if self.get_center_of_mass() < 0.6: break
                if abs(pm["total_angle"]) > 20: break
                cur_state = next_state
            if len(state_seq) == SEQUENCE_LENGTH:
                self.bc_buffer.push_sequence(state_seq, action_seq)

    def train_bc(self, epochs=200, lr=1e-3, weight_decay=0.0, clip_norm=None,
                patience=20, print_every=1, seq_batch_size=32, iters_per_epoch=10):
        self.actor.train()
        opt = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        best = float('inf'); no_improve = 0
        self.bc_losses = []

        for epoch in range(1, epochs+1):
            epoch_loss = 0.0
            for _ in range(iters_per_epoch):
                states, actions = self.bc_buffer.flatten_sample(batch_size=seq_batch_size)  # 32*35 ≈ 1120 样本
                states, actions = states.to(self.device), actions.to(self.device)

                # —— 归一化到 [-1,1] 空间训练（配合 Actor.tanh 输出）——
                mid = (self.actor.joint_min + self.actor.joint_max) * 0.5
                half = (self.actor.joint_max - self.actor.joint_min) * 0.5
                targets_norm = (torch.clamp(actions, self.actor.joint_min, self.actor.joint_max) - mid) / (half + 1e-6)

                pred = self.actor(states)             # 已在范围内
                pred_norm = (pred - mid) / (half + 1e-6)

                loss = F.mse_loss(pred_norm, targets_norm)
                opt.zero_grad(); loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), clip_norm)
                opt.step()

                epoch_loss += loss.item()

            val = epoch_loss / max(1, iters_per_epoch)
            self.bc_losses.append(val)
            if (epoch % print_every) == 0:
                print(f"[BC][Epoch {epoch:03d}] bc_loss = {val:.6f}")

            if val + 1e-6 < best:
                best = val; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[BC] Early stop (no improve {patience}). Best={best:.6f}")
                    break

    def get_center_of_mass(self):
        return self.get_translation()[2]
    
        # ====== BC 评测：只按摔倒或步数上限停止（无总位移检测） ======
    def eval_bc_walk(self,
                     max_steps: int = 3000,
                     com_fall_threshold: float = 0.6,
                     log_csv_path: str = None,
                     print_every: int = 50):
        """
        用当前 BC actor 连续行走，直到：
          - 摔倒 (COM < com_fall_threshold)，或
          - 达到 max_steps
        不做任何总位移/进展检测。

        返回：dict 总结统计；若提供 log_csv_path，将逐步写入 CSV。
        """
        import csv, time

        # 1) 准备
        self.actor.eval()
        dt = float(self.timeStep) / 1000.0  # 秒/步

        # 以“当前”为起点计距离
        self.pre_position = self.get_translation()
        self.start_position = self.get_translation()

        # 初始状态
        cur_state = torch.tensor(self.get_current_state(0), dtype=torch.float32, device=self.device)

        # 可选：逐步日志 CSV
        csv_writer = None
        f_csv = None
        if log_csv_path:
            f_csv = open(log_csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(f_csv, fieldnames=[
                "step", "dx", "total_distance", "yaw", "com"
            ])
            csv_writer.writeheader()

        reason = "max_steps_reached"
        total_distance = 0.0
        t0 = time.time()

        # 2) 主循环
        for step_idx in range(1, max_steps + 1):
            # 前向 & 动作
            with torch.no_grad():
                action = self.actor(cur_state.unsqueeze(0)).squeeze(0)

            # 施加 & 推进
            self.apply_action_to_robot(action)
            self.step(self.timeStep)

            # 读数（仅用于记录/显示，不做停止判据）
            pm = self.get_plane_movement()   # 会更新 self.pre_position
            dx = float(pm["distance_moved"])
            total_distance = float(pm["total_distance"])
            com = float(self.get_center_of_mass())
            yaw = float(self.get_yaw())

            if csv_writer:
                csv_writer.writerow({
                    "step": step_idx, "dx": dx,
                    "total_distance": total_distance,
                    "yaw": yaw, "com": com
                })

            if (print_every is not None) and (step_idx % int(print_every) == 0):
                print(f"[BC-EVAL] step={step_idx}  dx={dx:.4f}  total={total_distance:.3f}  yaw={yaw:.2f}  com={com:.3f}")

            # 摔倒即停止
            if com < float(com_fall_threshold):
                reason = f"fell(com={com:.3f}<{com_fall_threshold})"
                break

            # 下一步状态
            cur_state = torch.tensor(self.get_current_state(step_idx), dtype=torch.float32, device=self.device)

        if f_csv:
            f_csv.close()

        # 3) 汇总
        elapsed = time.time() - t0
        avg_dx_per_step = total_distance / max(1, step_idx)
        avg_speed = total_distance / max(1e-6, step_idx * dt)

        summary = {
            "steps": step_idx,
            "total_distance": total_distance,
            "avg_dx_per_step": avg_dx_per_step,
            "avg_speed_mps": avg_speed,
            "time_sec": elapsed,
            "stopped_by": reason
        }

        print("[BC-EVAL][summary] "
              f"steps={step_idx}  total={total_distance:.3f}  "
              f"avg_dx={avg_dx_per_step:.4f}/step  speed≈{avg_speed:.3f} m/s  "
              f"dt={dt:.3f}s  reason={reason}")

        return summary

    def eval_bc_walk_from_file(self, bc_path: str, **kwargs):
        """
        从文件加载 BC 权重并调用 eval_bc_walk（不进行总位移检测）。
        kwargs 会透传给 eval_bc_walk（如 max_steps、log_csv_path 等）。
        """
        self.load_bc_actor(bc_path, strict=True, eval_mode=True)
        print(f"[BC-EVAL] Loaded {bc_path}. Start evaluation…")
        return self.eval_bc_walk(**kwargs)


# —— 运行（在 Webots 里作为控制器使用）——
controller = Sprinter(auto_load_bc=True, bc_path="actor_bc.pth")  # 启动时自动尝试加载
controller.initialize()

# 如果你只想基于历史BC直接跑，不想采集&再训练，可以注释掉下面两行
controller.collect_bc_data(num_episodes=40)
controller.train_bc(epochs=100, lr=1e-3)
controller.eval_bc_walk(max_steps=2000, print_every=25)

# 训练后保存
controller.save_bc_actor("actor_bc.pth")
"""
controller.eval_bc_walk_from_file("actor_bc.pth",
                                 max_steps=3000,
                                 com_fall_threshold=0.6,
                                 log_csv_path="bc_eval_log.csv",
                                 print_every=50)
"""