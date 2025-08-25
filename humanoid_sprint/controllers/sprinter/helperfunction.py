import os
import csv
import json, re
from pathlib import Path
from datetime import datetime
from typing import List, Optional,Tuple

import numpy as np
import torch

# ====== 绘图（global_step 后缀） ======
from pathlib import Path


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
def save_checkpoint(host, filepath="check_points/ddpg_checkpoint.pth"):
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
    filepath="check_points/ddpg_checkpoint.pth",
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



def _save_line_chart(
    x,
    y1,
    y2,
    out_main: str,
    out_latest: Optional[str],
    title: str,
    x_label: str,
    y_limits: Optional[Tuple[float, float]] = None,   # ← 新：y 轴范围
    draw_zero_line: bool = True,                      # ← 新：画 y=0 参考线
    outliers: Optional[dict] = None                   # ← 新：标注溢出点
) -> None:
    import matplotlib
    try: matplotlib.use("Agg")
    except Exception: pass
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, y1, label="Qμ (mean Q)")
    plt.plot(x, y2, label="Rμ (mean reward)")

    if draw_zero_line:
        plt.axhline(0.0, linestyle="--", linewidth=1)

    if y_limits is not None:
        plt.ylim(y_limits[0], y_limits[1])

    # 溢出点（被裁出 y 轴范围的点）贴在边缘
    if outliers:
        yl = y_limits
        if yl is not None:
            if outliers.get("q_low_idx"):
                plt.scatter([x[i] for i in outliers["q_low_idx"]], [yl[0]]*len(outliers["q_low_idx"]),
                            marker="v", s=18, label="_nolegend_")
            if outliers.get("q_high_idx"):
                plt.scatter([x[i] for i in outliers["q_high_idx"]], [yl[1]]*len(outliers["q_high_idx"]),
                            marker="^", s=18, label="_nolegend_")
            if outliers.get("r_low_idx"):
                plt.scatter([x[i] for i in outliers["r_low_idx"]], [yl[0]]*len(outliers["r_low_idx"]),
                            marker="x", s=18, label="_nolegend_")
            if outliers.get("r_high_idx"):
                plt.scatter([x[i] for i in outliers["r_high_idx"]], [yl[1]]*len(outliers["r_high_idx"]),
                            marker="+", s=18, label="_nolegend_")

    plt.xlabel(x_label); plt.ylabel("Mean value"); plt.title(title)
    plt.legend(); plt.tight_layout()
    from pathlib import Path
    Path(out_main).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_main, dpi=160)
    if out_latest is not None:
        Path(out_latest).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_latest, dpi=160)
    plt.close(fig)
    print(f"[plot] saved -> {out_main}" + (f"  (also {out_latest})" if out_latest else ""))

def plot_q_reward_over_updates_gs(
    log_dir,
    qmu_list,
    rmu_list,
    global_step: int,
    base_filename: str = "summary_q_reward",
    title: str = "Qμ & Rμ over updates",
    last_n: Optional[int] = None,       # 只看最近 N 次；None=全部
    lower_quantile: float = 0.01,       # 只裁掉“最底部”这部分（默认1%）
    annotate_low_outliers: bool = True  # 被裁掉的低值点贴到底部
) -> Tuple[str, str]:
    import numpy as np

    n = min(len(qmu_list), len(rmu_list))
    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_filename).stem
    suf  = ".png"
    out_gs = str(plots_dir / f"{stem}_gs{int(global_step):06d}{suf}")
    out_latest = str(plots_dir / f"{stem}_latest{suf}")

    if n == 0:
        print(f"[plot] nothing to plot: q={len(qmu_list)}, r={len(rmu_list)}")
        return out_gs, out_latest

    start = max(0, n - last_n) if last_n is not None else 0
    yq = np.asarray(qmu_list[start:n], dtype=float)
    yr = np.asarray(rmu_list[start:n], dtype=float)
    x  = np.arange(1, n + 1)[start:]

    allv = np.concatenate([yq, yr], axis=0)
    lo = np.quantile(allv, lower_quantile)        # 仅下分位裁剪
    hi = float(np.max(allv))                      # 上界保留所有正极值
    pad = 0.05 * (hi - lo + 1e-9)
    ylim = (lo - pad, hi + pad)

    outliers = None
    if annotate_low_outliers:
        q_low_idx = [i for i, v in enumerate(yq) if v < ylim[0]]
        r_low_idx = [i for i, v in enumerate(yr) if v < ylim[0]]
        outliers = {"q_low_idx": q_low_idx, "q_high_idx": [],
                    "r_low_idx": r_low_idx, "r_high_idx": []}

    _save_line_chart(
        x, yq, yr, out_main=out_gs, out_latest=out_latest,
        title=title, x_label="Update step (train() call index)",
        y_limits=ylim, draw_zero_line=True, outliers=outliers
    )
    return out_gs, out_latest


def plot_q_reward_from_csv_gs(
    log_dir,
    global_step: int,
    metrics_name: str = "metrics.csv",
    base_filename: str = "summary_q_reward_from_csv",
    last_n: Optional[int] = None,
    lower_quantile: float = 0.01,       # 仅裁掉底部1%
    annotate_low_outliers: bool = True
) -> Tuple[str, str]:
    import csv, numpy as np

    csv_path = Path(log_dir) / metrics_name
    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_filename).stem
    suf  = ".png"
    out_gs = str(plots_dir / f"{stem}_gs{int(global_step):06d}{suf}")
    out_latest = str(plots_dir / f"{stem}_latest{suf}")

    if not csv_path.exists():
        print(f"[plot] metrics.csv not found: {csv_path}")
        return out_gs, out_latest

    pat_q = re.compile(r"Qμ=([-\d\.eE]+)")
    pat_r = re.compile(r"Rμ=([-\d\.eE]+)")
    q_vals, r_vals = [], []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            note = (row.get("note") or "")
            mq, mr = pat_q.search(note), pat_r.search(note)
            if mq and mr:
                try:
                    q_vals.append(float(mq.group(1)))
                    r_vals.append(float(mr.group(1)))
                except ValueError:
                    pass

    n = min(len(q_vals), len(r_vals))
    if n == 0:
        print(f"[plot] no Qμ/Rμ parsed from {csv_path}")
        return out_gs, out_latest

    start = max(0, n - last_n) if last_n is not None else 0
    yq = np.asarray(q_vals[start:n], dtype=float)
    yr = np.asarray(r_vals[start:n], dtype=float)
    x  = np.arange(1, n + 1)[start:]

    allv = np.concatenate([yq, yr], axis=0)
    lo = np.quantile(allv, lower_quantile)        # 仅下分位裁剪
    hi = float(np.max(allv))                      # 上界保留所有正极值
    pad = 0.05 * (hi - lo + 1e-9)
    ylim = (lo - pad, hi + pad)

    outliers = None
    if annotate_low_outliers:
        q_low_idx = [i for i, v in enumerate(yq) if v < ylim[0]]
        r_low_idx = [i for i, v in enumerate(yr) if v < ylim[0]]
        outliers = {"q_low_idx": q_low_idx, "q_high_idx": [],
                    "r_low_idx": r_low_idx, "r_high_idx": []}

    _save_line_chart(
        x, yq, yr, out_main=out_gs, out_latest=out_latest,
        title="Qμ & Rμ over updates (from CSV)",
        x_label="Update step (from CSV)",
        y_limits=ylim, draw_zero_line=True, outliers=outliers
    )
    return out_gs, out_latest
    _save_line_chart(
        x, yq, yr, out_main=out_gs, out_latest=out_latest,
        title="Qμ & Rμ over updates (from CSV)",
        x_label="Update step (from CSV)",
        y_limits=ylim, draw_zero_line=True, outliers=outliers
    )
    return out_gs, out_latest

def rotate_metrics_csv(log_dir: str) -> Optional[str]:
    """
    当 log_dir/metrics.csv 已存在时，把它重命名为 metrics_XXX.csv（编号递增），
    返回新的文件路径字符串；若原文件不存在则返回 None。
    """
    logp = Path(log_dir)
    logp.mkdir(parents=True, exist_ok=True)

    cur = logp / "metrics.csv"
    if not cur.exists():
        return None

    idx = 1
    while True:
        cand = logp / f"metrics_{idx:03d}.csv"
        if not cand.exists():
            cur.rename(cand)
            print(f"[logs] rotated metrics.csv -> {cand.name}")
            return str(cand)
        idx += 1