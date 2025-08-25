import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch


# ====== 动作读取 ======
def load_initial_actions_csv(
    filename: str,
    expected_dim: Optional[int] = None,
    to_radians: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """从 CSV 文件读取 initial actions，返回 tensor 列表。"""
    rows = []
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line:
                continue
            rows.append([float(x) for x in line])

    arr = np.asarray(rows, dtype=np.float32)

    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"initial_actions dim mismatch: got {arr.shape[1]}, expect {expected_dim}"
        )

    if to_radians:
        arr = np.deg2rad(arr)

    return [torch.tensor(a, dtype=dtype, device=device) for a in arr]


# ====== 日志记录 ======
def _prepare_logging(host, log_dir: str = "runs/walk_exp1", resume: bool = True) -> None:
    """初始化/恢复日志目录与计数器到 host（self）。"""
    host.log_dir = Path(log_dir)
    host.log_dir.mkdir(parents=True, exist_ok=True)
    host.metrics_csv = host.log_dir / "metrics.csv"
    host.meta_json = host.log_dir / "meta.json"

    host.global_step = getattr(host, "global_step", 0)
    host.ep_offset = getattr(host, "ep_offset", 0)
    host.best_reward = getattr(host, "best_reward", float("-inf"))

    if resume and host.meta_json.exists():
        meta = json.loads(host.meta_json.read_text(encoding="utf-8"))
        host.global_step = int(meta.get("global_step", host.global_step))
        host.ep_offset = int(meta.get("ep_offset", host.ep_offset))
        host.best_reward = float(meta.get("best_reward", host.best_reward))

    if not host.metrics_csv.exists():
        with host.metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "global_step",
                    "episode",
                    "type",
                    "actor_loss",
                    "critic_loss",
                    "reward",
                    "avg_recent_reward",
                    "fell",
                    "seq_len",
                    "note",
                    "time",
                ],
            )
            writer.writeheader()


def _append_metric(host, row: dict) -> None:
    """追加一行指标到 metrics.csv。"""
    defaults = {
        "global_step": getattr(host, "global_step", 0),
        "episode": "",
        "type": "",
        "actor_loss": "",
        "critic_loss": "",
        "reward": "",
        "avg_recent_reward": "",
        "fell": "",
        "seq_len": "",
        "note": "",
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    out = {**defaults, **row}

    with host.metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "episode",
                "type",
                "actor_loss",
                "critic_loss",
                "reward",
                "avg_recent_reward",
                "fell",
                "seq_len",
                "note",
                "time",
            ],
        )
        writer.writerow(out)


def _save_meta(host) -> None:
    """保存 host 的 meta 信息到文件。"""
    meta = {
        "global_step": int(getattr(host, "global_step", 0)),
        "ep_offset": int(getattr(host, "ep_offset", 0)),
        "best_reward": float(getattr(host, "best_reward", float("-inf"))),
    }
    host.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ====== 模型保存/加载 ======
def save_checkpoint(host, filepath="ddpg_checkpoint.pth"):
    checkpoint = {
        "actor_state_dict": host.actor.state_dict(),
        "critic_state_dict": host.critic.state_dict(),
        "actor_target_state_dict": host.actor_target.state_dict(),
        "critic_target_state_dict": host.critic_target.state_dict(),
        "actor_optimizer": host.actor_optimizer.state_dict(),
        "critic_optimizer": host.critic_optimizer.state_dict(),
        "train_logs": getattr(host, "_train_logs", None),
    }
    torch.save(checkpoint, filepath)
    print(f"Saved DDPG checkpoint (with target networks) to {filepath}")


def load_checkpoint(
    host,
    filepath="ddpg_checkpoint.pth",
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
):
    if not os.path.exists(filepath):
        print(f" Checkpoint not found at {filepath}")
        return

    checkpoint = torch.load(filepath, map_location="cpu")
    name, _ = next(host.critic.named_parameters())
    print(
        f" Before load: critic parameter '{name}' first 5 vals =",
        host.critic.state_dict()[name].view(-1)[:5],
    )

    host.actor.load_state_dict(checkpoint["actor_state_dict"])
    host.critic.load_state_dict(checkpoint["critic_state_dict"])
    host.actor.to(host.device)
    host.critic.to(host.device)

    if "actor_target_state_dict" in checkpoint:
        host.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        host.actor_target.to(host.device)
        print(" Loaded actor_target from checkpoint.")
    else:
        print("actor_target not found in checkpoint.")

    if "critic_target_state_dict" in checkpoint:
        host.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        host.critic_target.to(host.device)
        print(" Loaded critic_target from checkpoint.")
    else:
        print(" critic_target not found in checkpoint.")

    print(
        f"▶ After load: critic parameter '{name}' first 5 vals =",
        host.critic.state_dict()[name].view(-1)[:5],
    )
    print(
        f"▶ After load: critic_target parameter '{name}' first 5 vals =",
        host.critic_target.state_dict()[name].view(-1)[:5],
    )

    if "actor_optimizer" in checkpoint:
        host.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint:
        host.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    for opt in (host.actor_optimizer, host.critic_optimizer):
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(host.device)

    if actor_lr is not None:
        for pg in host.actor_optimizer.param_groups:
            pg["lr"] = actor_lr
        print(f" Actor optimizer learning rate set to {actor_lr}")
    if critic_lr is not None:
        for pg in host.critic_optimizer.param_groups:
            pg["lr"] = critic_lr
        print(f" Critic optimizer learning rate set to {critic_lr}")

    print(f" Loaded checkpoint from {filepath} onto {host.device}")


def load_bc(host, filepath="actor_bc.pth"):
    if not os.path.exists(filepath):
        print(f" Behavior cloning model not found at {filepath}")
        return

    state_dict = torch.load(filepath, map_location="cpu")
    host.actor.load_state_dict(state_dict)
    host.actor_target.load_state_dict(state_dict)  # 同步 target actor
    host.actor.to(host.device)
    host.actor_target.to(host.device)

    print(
        f"Loaded behavior cloning model into actor and target actor from '{filepath}'"
    )
