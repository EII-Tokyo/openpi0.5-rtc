#!/usr/bin/env python3
"""Generate evidence-bound figures for the ALOHA first-year report.

Every figure below answers one named report question. The script deliberately
avoids calculating task success rates because no formal trial ledger exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ART / "plot_data"
FIG = ROOT / "figures"
BUILD = ROOT / "build"
FIG.mkdir(parents=True, exist_ok=True)

NAVY = "#142B4A"
BLUE = "#2474B5"
CYAN = "#38B6C4"
TEAL = "#168B83"
AMBER = "#E69F35"
RED = "#C6534C"
GREEN = "#4C956C"
TOP_CAMERA_TARGET = (856, 241)
TERMINAL_BOX_GAP = .050
GRAY = "#667085"
LIGHT = "#F2F6F8"
INK = "#182230"

mpl.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D0D5DD",
        "grid.color": "#E6EBEF",
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_json(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def rounded_box(ax, xy, width, height, title, body, color=BLUE, alpha=1.0, size=10):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=1.6,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center",
            fontsize=size, weight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.33, body, ha="center", va="center",
            fontsize=size - 1, color=INK, linespacing=1.35)
    return patch


def figure_data_scope() -> None:
    hf = load_json("hf_training_dataset_audit.json")["totals"]
    dc = load_json("datacenter_aloha_audit.json")["project_totals"]
    values = [
        dc["declared_duration_sec"] / 3600,
        hf["unique_duration_sec_at_declared_fps"] / 3600,
        hf["trainable_frames"] / 50 / 3600,
    ]
    labels = ["数据平台有效 ALOHA 资产", "正式模型唯一训练子集", "过滤后实际可训练部分"]
    colors = [NAVY, BLUE, CYAN]
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.56)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.2f} 小时",
                va="center", weight="bold", color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("按 50 帧/秒换算的视频时长（小时）")
    ax.set_title("从数据资产到正式训练输入：三层口径不能混用", loc="left")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(
        0.99,
        0.05,
        "平台总资产：51 个项目、2,413 条轨迹、2,907,804 帧\n"
        "正式训练子集：25 个唯一仓库、1,051 条轨迹、879,852 帧\n"
        "过滤后：844,102 帧",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=.5", fc=LIGHT, ec="none"),
    )
    save(fig, "data_scope")


def figure_sampling_exposure() -> None:
    totals = load_json("hf_training_dataset_audit.json")["totals"]
    labels = ["唯一轨迹", "训练采样暴露"]
    values = [totals["unique_episodes"], totals["weighted_episode_exposure"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(labels, values, width=0.56, color=[BLUE, AMBER])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 28, f"{v:,}", ha="center",
                weight="bold", fontsize=13, color=INK)
    ax.set_ylim(0, 1700)
    ax.set_ylabel("轨迹等效数量")
    ax.set_title("重复采样提高重点场景出现机会，但不会产生新轨迹", loc="left")
    ax.grid(axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(1, 430, "其中“自由旋转瓶盖”数据\n在部署训练中重复 5 次",
            ha="center", color=AMBER, fontsize=9,
            bbox=dict(boxstyle="round,pad=.4", fc="#FFF7E7", ec=AMBER))
    save(fig, "sampling_exposure")


def figure_episode_lengths() -> None:
    df = pd.read_csv(DATA / "baseline_training_episode_lengths.csv")
    seconds = df["length"] / 50.0
    q = seconds.quantile([0.1, 0.5, 0.9])
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bins = np.linspace(0, max(30, seconds.quantile(0.99)), 30)
    ax.hist(seconds.clip(upper=bins[-1]), bins=bins, color=BLUE, alpha=0.88, edgecolor="white")
    ax.axvline(q.loc[0.5], color=AMBER, linewidth=2.2, label=f"中位数 {q.loc[0.5]:.1f} 秒")
    ax.axvspan(q.loc[0.1], q.loc[0.9], color=CYAN, alpha=0.13,
               label=f"中间 80%：{q.loc[0.1]:.1f}–{q.loc[0.9]:.1f} 秒")
    ax.set_xlabel("单条示教时长（秒）")
    ax.set_ylabel("轨迹数量")
    ax.set_title("1,051 条正式训练轨迹的长度分布", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "baseline_episode_length_distribution")


def figure_training_loss() -> None:
    df = pd.read_csv(DATA / "baseline_training_history.csv").sort_values("_step")
    df["smooth"] = df["loss"].rolling(101, center=True, min_periods=5).median()
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.plot(df["_step"], df["loss"], color=BLUE, alpha=0.16, linewidth=0.7, label="每次记录")
    ax.plot(df["_step"], df["smooth"], color=NAVY, linewidth=2.2, label="滚动中位趋势")
    ax.axvline(19000, color=AMBER, linewidth=2, linestyle="--", label="现场部署保存点：19,000")
    ax.axvline(df["_step"].max(), color=GRAY, linewidth=1.3, linestyle=":", label="训练历史终点：59,990")
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("示教动作拟合误差（对数坐标）")
    ax.set_title("正式模型确实持续学习示教动作，但没有同步验证集曲线", loc="left")
    ax.legend(frameon=False, ncol=2)
    ax.grid(which="both", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.985,
        0.93,
        "首值 0.0677\n部署点约在训练中段\n末值 0.000671",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=.5", fc=LIGHT, ec="none"),
    )
    save(fig, "baseline_training_loss")


def figure_experiment_funnel() -> None:
    inventory = load_json("wandb_experiment_inventory.json")["family_summary"]
    stages = ["运行尝试", "留下训练历史", "达到 1 万步", "达到 2.5 万步", "进程标记结束"]
    keys = ["run_attempts", "runs_with_any_history", "runs_reaching_10000_steps",
            "runs_reaching_25000_steps"]
    families = [
        ("瓶子分拣基础模型", inventory["baseline_bottle_sorting"], BLUE),
        ("冲洗/插入探索", inventory["rinse_or_insertion_exploration"], TEAL),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    y = np.arange(len(stages))
    height = 0.34
    for offset, (label, stats, color) in zip([-height / 2, height / 2], families):
        values = [stats[k] for k in keys] + [stats["states"].get("finished", 0)]
        ax.barh(y + offset, values, height=height, label=label, color=color)
        for yy, value in zip(y + offset, values):
            ax.text(value + 0.28, yy, str(value), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlabel("运行记录数量")
    ax.set_xlim(0, 25)
    ax.set_title("训练试验漏斗：41 次工程尝试逐步收敛", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "experiment_funnel")


def figure_condition_coverage() -> None:
    cov = load_json("dataset_condition_coverage.json")["category_summary"]
    order = [
        ("方向变化", "direction_named"),
        ("无盖瓶", "no_cap_named"),
        ("回到初始位", "return_home_named"),
        ("含水瓶", "water_named"),
        ("翻转瓶身", "turn_over_named"),
        ("瓶盖自由旋转", "free_spinning_named"),
    ]
    labels = [x[0] for x in order]
    episodes = [cov[x[1]]["episodes"] for x in order]
    weighted = [cov[x[1]]["deployed_weighted_episodes"] for x in order]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.barh(y, weighted, color="#DCECF4", height=.62, label="部署训练中的采样暴露")
    ax.barh(y, episodes, color=BLUE, height=.36, label="真实独立轨迹")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("轨迹数量 / 等效采样数量")
    ax.set_title("训练数据主动覆盖多种条件，但覆盖并不均衡", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yy, (real, exp) in enumerate(zip(episodes, weighted)):
        ax.text(real + 5, yy, f"{real}", va="center", color=NAVY, fontsize=8.5)
        if exp > real:
            ax.text(exp + 5, yy, f"采样到 {exp}", va="center", color=AMBER, fontsize=8.5)
    save(fig, "condition_coverage")


def figure_prompt_mismatch() -> None:
    prompt = load_json("dataset_condition_coverage.json")["prompt_summary"]
    cross = load_json("dataset_condition_coverage.json")["no_cap_prompt_cross_check"]
    labels = ["无条件“拧盖”短指令", "长指令但仍无条件拧盖", "明确“有盖才拧”"]
    values = [
        prompt["short_unconditional_unscrew"]["episodes"],
        prompt["long_but_unconditional_cap_step"]["episodes"],
        prompt["conditional_on_cap_presence"]["episodes"],
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.7), gridspec_kw={"width_ratios": [1.4, 1]})
    colors = [BLUE, AMBER, GREEN]
    bars = ax1.barh(labels[::-1], values[::-1], color=colors[::-1], height=.56)
    for b, v in zip(bars, values[::-1]):
        ax1.text(v + 18, b.get_y() + b.get_height() / 2, f"{v:,} 条", va="center", weight="bold")
    ax1.set_xlim(0, 1050)
    ax1.set_xlabel("轨迹数量")
    ax1.set_title("任务指令的条件覆盖", loc="left")
    ax1.grid(axis="x")
    ax1.spines[["top", "right", "left"]].set_visible(False)

    ax2.axis("off")
    rounded_box(ax2, (.08, .55), .84, .27, "无盖数据交叉检查",
                f"{cross['repositories']} 个仓库 / {cross['episodes']} 条轨迹", RED, size=11)
    rounded_box(ax2, (.08, .15), .84, .27, "其中明确“无盖则跳过拧盖”",
                "0 个仓库", RED, size=11)
    ax2.annotate("", xy=(.5, .43), xytext=(.5, .55),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax2.text(.5, .06, "这与现场“无盖仍空拧”现象一致，\n但仍需配对实验才能确认因果。",
             ha="center", va="center", fontsize=9, color=GRAY)
    save(fig, "prompt_condition_gap")


def figure_attention_share() -> None:
    df = pd.read_csv(DATA / "attention_camera_share.csv")
    first_mask = df.groupby("run_id").cumcount() == 0
    df = df[~first_mask].copy()
    columns = ["cam_high_share", "cam_left_wrist_share", "cam_right_wrist_share"]
    labels = ["顶部总览", "左腕近景", "右腕近景"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    parts = ax.violinplot([df[c] * 100 for c in columns], showmeans=False, showmedians=True,
                         widths=.72)
    for body, color in zip(parts["bodies"], [BLUE, TEAL, AMBER]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(.82)
    parts["cmedians"].set_color(NAVY)
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel("三路视野中的注意力份额（%）")
    ax.set_title("8,223 个真实记录：模型更依赖靠近操作点的腕部视野", loc="left")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    medians = [(df[c] * 100).median() for c in columns]
    for x, med in enumerate(medians, start=1):
        ax.text(x, med + 1.2, f"中位 {med:.1f}%", ha="center", fontsize=8.5, weight="bold")
    save(fig, "attention_camera_share")


def figure_rlt_eval() -> None:
    df = pd.read_csv(DATA / "round_eval.csv")
    last = df.sort_values(["round", "step"]).groupby("round", as_index=False).tail(1)
    x = last["round"].to_numpy()
    mae = last["val_actor_mae"].to_numpy()
    critic = last["val_critic_loss"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(9.6, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(x, mae, marker="o", linewidth=2.2, color=BLUE, label="验证动作误差")
    ax2.plot(x, critic, marker="s", linewidth=1.8, color=AMBER, label="价值判断误差")
    ax1.set_xlabel("连续研究轮次")
    ax1.set_ylabel("验证动作误差（越低越好）", color=BLUE)
    ax2.set_ylabel("价值判断误差（越低越好）", color=AMBER)
    ax1.set_xticks(x)
    ax1.grid(axis="y")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_title("强化学习方向已形成可训练闭环，但当前证据仍停留在离线验证", loc="left")
    improvement = (mae[0] - mae[-1]) / mae[0] * 100
    ax1.text(0.02, .08, f"动作误差从 {mae[0]:.3f} 降至 {mae[-1]:.3f}\n相对下降约 {improvement:.1f}%",
             transform=ax1.transAxes, color=BLUE, fontsize=9,
             bbox=dict(boxstyle="round,pad=.45", fc=LIGHT, ec="none"))
    ax1.text(.98, .08, "没有同条件基础模型对照\n没有真机成功率记录",
             transform=ax1.transAxes, ha="right", color=RED, fontsize=9,
             bbox=dict(boxstyle="round,pad=.45", fc="#FFF2F0", ec="none"))
    save(fig, "rlt_offline_validation")


def figure_evidence_grade() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (.03, .29), .27, .48, "A级｜可复核事实",
                "正式保存模型\n训练历史与配置\n1,051 条训练轨迹\n8,223 份注意力记录", GREEN, size=11)
    rounded_box(ax, (.365, .29), .27, .48, "B级｜现场观测估计",
                "完整分拣循环\n连续工作 >1 小时\n平均约 2 瓶/分钟\n空拧、倒抓现象", AMBER, size=11)
    rounded_box(ax, (.70, .29), .27, .48, "C级｜尚缺正式测量",
                "任务成功率\n分条件成功率\n失败类型频次\n吞吐量置信区间", RED, size=11)
    for x0, x1 in [(.30, .365), (.635, .70)]:
        ax.add_patch(FancyArrowPatch((x0, .53), (x1, .53), arrowstyle="-|>",
                                     mutation_scale=15, lw=1.8, color=GRAY))
    save(fig, "evidence_grade")


def figure_model_dataflow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (.025, .56, .15, .25, "三路视野", "顶部总览\n左腕近景\n右腕近景", BLUE),
        (.025, .17, .15, .25, "机器人状态", "双臂关节\n双夹爪位置\n共 14 个量", TEAL),
        (.23, .36, .18, .28, "统一理解", "把画面、动作状态\n和任务指令放到\n同一表示中", NAVY),
        (.47, .36, .18, .28, "动作生成", "从随机动作开始\n分 10 次修正\n得到未来动作", AMBER),
        (.71, .36, .15, .28, "动作片段", "一次预测未来\n50 个控制时刻\n共 14 个量", BLUE),
        (.86 + TERMINAL_BOX_GAP, .36, .065, .28, "双臂", "50 Hz\n连续执行", GREEN),
    ]
    for x, y, w, h, title, body, color in boxes:
        rounded_box(ax, (x, y), w, h, title, body, color, size=10)
    arrows = [
        ((.175, .69), (.23, .54)),
        ((.175, .30), (.23, .46)),
        ((.41, .50), (.47, .50)),
        ((.65, .50), (.71, .50)),
        ((.868, .50), (.86 + TERMINAL_BOX_GAP - .008, .50)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                     color=GRAY, lw=1.8, connectionstyle="arc3,rad=0"))
    ax.text(.55, .77, "训练：让预测的“去噪方向”靠近示教动作的真实方向",
            ha="center", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=.45", fc=LIGHT, ec="none"))
    ax.set_title("正式部署模型：看三路画面，连续生成双臂动作片段", loc="left", pad=12)
    save(fig, "model_dataflow")


def figure_task_timeline() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1", "发现与接近", "多瓶随机位置"),
        ("2", "左爪取瓶", "抓稳并抬起"),
        ("3", "右爪找盖", "两臂靠近接触"),
        ("4", "协同旋拧", "左稳瓶、右旋盖"),
        ("5", "分离与确认", "瓶盖脱离"),
        ("6", "分类投放", "瓶身/瓶盖进不同盒"),
        ("7", "回到下一轮", "长程循环"),
    ]
    xs = np.linspace(.07, .93, len(stages))
    ax.plot(xs, [.54] * len(xs), color="#C9D3DC", linewidth=3, zorder=0)
    for i, (x, (num, title, body)) in enumerate(zip(xs, stages)):
        color = [BLUE, TEAL, AMBER, RED, AMBER, GREEN, NAVY][i]
        ax.scatter([x], [.54], s=850, color=color, edgecolor="white", linewidth=2, zorder=2)
        ax.text(x, .54, num, ha="center", va="center", color="white", weight="bold", fontsize=12)
        ax.text(x, .35 if i % 2 == 0 else .76, title, ha="center", weight="bold", color=color, fontsize=9)
        ax.text(x, .26 if i % 2 == 0 else .85, body, ha="center", color=GRAY, fontsize=8)
    ax.set_title("一个瓶子不是一次抓取，而是七阶段双臂长流程", loc="left")
    save(fig, "task_timeline")


def figure_software_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (.03, .40, .16, .30, "专用采集软件", "连续采集\n脚踏/键盘触发\n丢弃重采\n安全检查", BLUE),
        (.235, .40, .16, .30, "数据编辑中心", "51 个有效项目\n浏览与标签\n清洗与生成\n任务跟踪", TEAL),
        (.44, .40, .16, .30, "数据发布与版本", "多视角视频\n轨迹与任务指令\n固定版本\n可追溯来源", NAVY),
        (.645, .40, .16, .30, "模型训练", "方案试验\n质量检查\n重点条件采样\n保存模型", AMBER),
        (.85, .40, .12, .30, "现场运行", "三路视觉\n双臂动作\n连续重规划", GREEN),
    ]
    for x, y, w, h, title, body, color in items:
        rounded_box(ax, (x, y), w, h, title, body, color, size=10)
    for i in range(len(items) - 1):
        x0 = items[i][0] + items[i][2]
        x1 = items[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0, .55), (x1, .55), arrowstyle="-|>",
                                     mutation_scale=14, color=GRAY, lw=1.8))
    ax.text(.315, .23, "数据返工与重采", ha="center", fontsize=9, color=RED)
    ax.add_patch(FancyArrowPatch((.44, .37), (.19, .37), arrowstyle="-|>",
                                 mutation_scale=13, color=RED, lw=1.3,
                                 connectionstyle="arc3,rad=-.25"))
    ax.text(.745, .23, "失败条件回流", ha="center", fontsize=9, color=RED)
    ax.add_patch(FancyArrowPatch((.89, .36), (.60, .36), arrowstyle="-|>",
                                 mutation_scale=13, color=RED, lw=1.3,
                                 connectionstyle="arc3,rad=-.23"))
    ax.set_title("第一年度建立的不只是模型，而是一条可重复的数据—训练—部署闭环", loc="left")
    save(fig, "software_data_pipeline")


def figure_engineering_workload() -> None:
    dataset = load_json("dataset_statistics.json")
    runs = load_json("wandb_experiment_inventory.json")["summary"]["all_selected"]
    attention = load_json("attention_audit.json")
    checkpoint = load_json("checkpoint_metadata.json")
    rlt = load_json("rlt_dataset_statistics.json")
    platform = dataset["platform_aloha_assets"]
    deployed = dataset["deployed_training_dataset"]

    cards = [
        ("数据资产", f"{platform['projects']} 个项目", f"{platform['declared_episodes']:,} 条轨迹｜约 16.15 小时", NAVY),
        ("正式训练数据", f"{deployed['unique_episodes']:,} 条轨迹", f"{deployed['unique_frames']:,} 帧｜25 个固定版本", BLUE),
        ("训练探索", f"{runs['run_attempts']} 次尝试", f"{runs['unique_config_names']} 组配置｜6 种批量规模", TEAL),
        ("正式训练", "59,990 步", "6,059 条记录｜约 51.4 小时", AMBER),
        ("部署模型", "8.38 亿参数", f"第 {checkpoint['directory_step']:,} 步｜三路视觉", GREEN),
        ("视觉审查", f"{attention['total_samples']:,} 个样本", f"{attention['manifest_count']} 份清单｜三路视野", CYAN),
        ("强化学习研究", f"{rlt['raw_dataset']['episodes']} 条轨迹", f"{rlt['training_replay']['transitions']['sum']:,} 个片段｜5 个轮次", RED),
    ]
    positions = [
        (.02, .57), (.265, .57), (.51, .57), (.755, .57),
        (.14, .16), (.39, .16), (.64, .16),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for (title, value, detail, color), (x, y) in zip(cards, positions):
        patch = FancyBboxPatch(
            (x, y), .225, .28,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white", edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (x, y + .225), .225, .055,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=color, edgecolor=color, linewidth=0,
            )
        )
        ax.text(x + .1125, y + .252, title, ha="center", va="center",
                color="white", fontsize=10, weight="bold")
        ax.text(x + .1125, y + .145, value, ha="center", va="center",
                color=color, fontsize=15, weight="bold")
        ax.text(x + .1125, y + .065, detail, ha="center", va="center",
                color=INK, fontsize=8.5)
    ax.set_title("从数据生产到现场部署：第一年度形成 7 类可核验工程资产",
                 loc="left", pad=12)
    save(fig, "engineering_workload_dashboard")
    (ART / "engineering_workload.json").write_text(
        json.dumps(
            {
                "cards": [
                    {"title": title, "value": value, "detail": detail}
                    for title, value, detail, _ in cards
                ],
                "source_artifacts": [
                    "artifacts/dataset_statistics.json",
                    "artifacts/wandb_experiment_inventory.json",
                    "artifacts/attention_audit.json",
                    "artifacts/checkpoint_metadata.json",
                    "artifacts/rlt_dataset_statistics.json",
                ],
                "interpretation_limit": "The cards use different units and must not be summed.",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def figure_roadmap() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        ("P0", "让结果可信", "固定测试条件｜逐瓶计数｜自动成功判定｜记录失败类型", RED),
        ("P1", "修复基础能力", "补采“无盖跳过”与正反方向数据｜配对评测｜稳定抓取与旋拧", AMBER),
        ("P2", "用强化学习优化", "先建立同条件基础模型对照｜离线筛选｜小规模安全真机验证", BLUE),
        ("P3", "推进插管清洗", "数字孪生标定｜毫米级几何测量｜视觉与接触协同｜分阶段验收", TEAL),
    ]
    y_positions = [.76, .56, .36, .16]
    for (priority, title, body, color), y in zip(rows, y_positions):
        ax.add_patch(FancyBboxPatch((.03, y - .07), .10, .14,
                                    boxstyle="round,pad=.01,rounding_size=.03",
                                    fc=color, ec="none"))
        ax.text(.08, y, priority, color="white", ha="center", va="center",
                fontsize=13, weight="bold")
        ax.text(.16, y + .025, title, va="center", fontsize=11, weight="bold", color=color)
        ax.text(.16, y - .035, body, va="center", fontsize=9.3, color=INK)
        if y != y_positions[-1]:
            ax.add_patch(FancyArrowPatch((.08, y - .08), (.08, y - .12),
                                         arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.4))
    ax.text(.99, .93, "先测清楚 → 再补数据 → 再优化 → 最后挑战高精度插入",
            ha="right", fontsize=10, color=GRAY)
    ax.set_title("未来一年路线：每一步都由当前证据缺口触发", loc="left")
    save(fig, "next_year_roadmap")


def annotated_photo() -> None:
    src = Path("/home/eii/Downloads/aloha-home.jpg")
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font = ImageFont.truetype(font_path, 31)
    bold = ImageFont.truetype(bold_path, 33)
    small = ImageFont.truetype(font_path, 25)

    def label(text, box, point, color):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255, 226),
                               outline=color + (255,), width=4)
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=bold,
                  fill=color + (255,), anchor="mm", align="center")
        start = ((x0 + x1) / 2, y1 if point[1] > y1 else y0)
        draw.line([start, point], fill=color + (255,), width=7)
        r = 10
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r),
                     fill=color + (255,))

    label("左侧工作臂", (340, 430, 590, 505), (520, 700), (36, 116, 181))
    label("右侧工作臂", (1110, 410, 1375, 485), (1190, 700), (22, 139, 131))
    label("顶部总览相机", (690, 330, 1010, 405), TOP_CAMERA_TARGET, (230, 159, 53))
    label("操作者示教臂", (65, 920, 330, 995), (285, 850), (102, 112, 133))
    label("操作者示教臂", (1370, 900, 1635, 975), (1440, 830), (102, 112, 133))
    draw.rounded_rectangle((475, 750, 1230, 930), radius=25, outline=(198, 83, 76, 230),
                           width=6, fill=(198, 83, 76, 28))
    draw.rounded_rectangle((650, 825, 1050, 900), radius=16, fill=(255, 255, 255, 220))
    draw.text((850, 862), "机器人实际操作区域", font=bold, fill=(198, 83, 76, 255),
              anchor="mm")
    draw.rounded_rectangle((25, 25, 670, 100), radius=15, fill=(20, 43, 74, 220))
    draw.text((48, 62), "正式设备照片｜静止休眠状态（非任务执行画面）",
              font=small, fill="white", anchor="lm")
    image.save(FIG / "aloha_formal_photo_annotated.png", quality=95)
    (ART / "official_photo_provenance.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "source_dimensions": list(Image.open(src).size),
                "output": "figures/aloha_formal_photo_annotated.png",
                "allowed_edits": ["labels", "arrows", "operation-region overlay"],
                "prohibition": "Not an execution or success image; no scene content was synthesized.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def training_keyframe_grid() -> None:
    categories = [
        ("普通完整示教", "ordinary_full_task__episode_0009"),
        ("多方向瓶子", "direction__episode_0052"),
        ("无盖条件", "no_cap__episode_0006"),
        ("瓶身翻转条件", "turn_over__episode_0022"),
    ]
    fig, axes = plt.subplots(len(categories), 3, figsize=(10.6, 9.1))
    for row, (title, prefix) in enumerate(categories):
        for col, pct in enumerate([20, 50, 80]):
            path = BUILD / "hf_training_keyframes" / f"{prefix}__p{pct}.png"
            axes[row, col].imshow(Image.open(path))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(["前段", "中段", "后段"][col], fontsize=10, weight="bold")
        axes[row, 0].text(-.03, .5, title, transform=axes[row, 0].transAxes,
                          ha="right", va="center", rotation=90, fontsize=10,
                          weight="bold", color=NAVY)
    fig.suptitle("真实训练示教关键帧：场景多样性存在，但这些不是自主测试结果",
                 fontsize=14, weight="bold", x=.02, ha="left")
    fig.tight_layout(rect=(.03, .01, 1, .96), h_pad=.45, w_pad=.25)
    fig.savefig(FIG / "training_demonstration_keyframes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def attention_example() -> None:
    path = BUILD / "attention_review/20260729-043606/sample_000730/overview.jpg"
    img = Image.open(path).convert("RGB")
    # Remove the original internal English labels while retaining the real heatmap.
    crop = img.crop((0, 48, img.width, img.height))
    shares = [22.3, 30.1, 47.5]
    fig, ax = plt.subplots(figsize=(11.0, 3.3))
    ax.imshow(crop)
    ax.axis("off")
    segment = crop.width / 3
    for i, (name, share) in enumerate(zip(["顶部总览", "左腕近景", "右腕近景"], shares)):
        ax.text((i + .5) * segment, 18, f"{name}｜本样例份额 {share:.1f}%",
                ha="center", va="top", color="white", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=.35", fc=(0.05, .12, .20, .78), ec="none"))
    fig.suptitle("真实注意力样例：彩色热区表示模型在生成动作时更集中查看的位置",
                 fontsize=13, weight="bold", x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(FIG / "attention_real_example.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_plot_manifest() -> None:
    records = [
        ("data_scope", "数据资产口径", "总资产与正式训练子集的区别"),
        ("sampling_exposure", "重点采样", "真实轨迹与采样暴露的区别"),
        ("baseline_episode_length_distribution", "示教长度", "轨迹长度是否单一"),
        ("baseline_training_loss", "训练真实性", "训练误差下降但无验证成功率"),
        ("experiment_funnel", "工程工作量", "运行尝试与可用训练的区别"),
        ("condition_coverage", "条件覆盖", "方向、无盖、翻转等数据分布"),
        ("prompt_condition_gap", "空拧风险", "无盖数据仍使用无条件拧盖指令"),
        ("attention_camera_share", "视觉利用", "三路视野的注意力份额"),
        ("attention_real_example", "真实注意力", "展示真实热区与解释边界"),
        ("rlt_offline_validation", "强化学习探索", "只证明离线训练闭环"),
        ("evidence_grade", "结论可信度", "区分核验事实、现场估计和缺失指标"),
        ("model_dataflow", "方法", "观察到双臂动作的完整数据流"),
        ("task_timeline", "任务难度", "七阶段长流程与误差累积"),
        ("software_data_pipeline", "平台工作量", "采集、编辑、发布、训练、部署闭环"),
        ("engineering_workload_dashboard", "工程工作强度", "七类可核验工程资产"),
        ("next_year_roadmap", "未来一年", "优先级与依赖关系"),
        ("training_demonstration_keyframes", "真实训练图像", "展示数据覆盖而非自主成功"),
        ("aloha_formal_photo_annotated", "正式设备", "静止状态设备组成"),
    ]
    out = [
        {"figure": name, "question": question, "purpose": purpose}
        for name, question, purpose in records
    ]
    (ART / "plot_manifest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    figure_data_scope()
    figure_sampling_exposure()
    figure_episode_lengths()
    figure_training_loss()
    figure_experiment_funnel()
    figure_condition_coverage()
    figure_prompt_mismatch()
    figure_attention_share()
    figure_rlt_eval()
    figure_evidence_grade()
    figure_model_dataflow()
    figure_task_timeline()
    figure_software_pipeline()
    figure_engineering_workload()
    figure_roadmap()
    annotated_photo()
    training_keyframe_grid()
    attention_example()
    write_plot_manifest()
    print(f"Generated {len(json.loads((ART / 'plot_manifest.json').read_text()))} purposeful figures.")


if __name__ == "__main__":
    main()
