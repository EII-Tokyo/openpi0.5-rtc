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
    labels = ["データ基盤の有効 ALOHA 資産", "本番運用モデルの一意な学習部分集合", "フィルタ後の実学習対象"]
    colors = [NAVY, BLUE, CYAN]
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.56)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.2f} 時間",
                va="center", weight="bold", color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("50フレーム/秒換算の映像時間（時間）")
    ax.set_title("データ資産から本番学習入力まで：三つの集計範囲は区別する", loc="left")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(
        0.99,
        0.05,
        "基盤全資産：51プロジェクト、2,413軌跡、2,907,804フレーム\n"
        "本番学習部分集合：25件の重複しないデータ保管単位、1,051軌跡、879,852フレーム\n"
        "フィルタ後：844,102フレーム",
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
    labels = ["一意な軌跡", "学習時の等価サンプリング量"]
    values = [totals["unique_episodes"], totals["weighted_episode_exposure"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(labels, values, width=0.56, color=[BLUE, AMBER])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 28, f"{v:,}", ha="center",
                weight="bold", fontsize=13, color=INK)
    ax.set_ylim(0, 1700)
    ax.set_ylabel("軌跡換算数")
    ax.set_title("反復サンプリングは重点場面の出現機会を増やすが、新規軌跡ではない", loc="left")
    ax.grid(axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(1, 430, "うち「自由回転するキャップ」データは\n実運用モデル学習で5回反復",
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
    ax.axvline(q.loc[0.5], color=AMBER, linewidth=2.2, label=f"中央値 {q.loc[0.5]:.1f} 秒")
    ax.axvspan(q.loc[0.1], q.loc[0.9], color=CYAN, alpha=0.13,
               label=f"中央80%：{q.loc[0.1]:.1f}–{q.loc[0.9]:.1f} 秒")
    ax.set_xlabel("1軌跡当たりの教示時間（秒）")
    ax.set_ylabel("軌跡数")
    ax.set_title("本番学習1,051軌跡の長さ分布", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "baseline_episode_length_distribution")


def figure_training_loss() -> None:
    df = pd.read_csv(DATA / "baseline_training_history.csv").sort_values("_step")
    df["smooth"] = df["loss"].rolling(101, center=True, min_periods=5).median()
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.plot(df["_step"], df["loss"], color=BLUE, alpha=0.16, linewidth=0.7, label="各記録")
    ax.plot(df["_step"], df["smooth"], color=NAVY, linewidth=2.2, label="移動中央値")
    ax.axvline(19000, color=AMBER, linewidth=2, linestyle="--", label="現場運用モデル：19,000")
    ax.axvline(df["_step"].max(), color=GRAY, linewidth=1.3, linestyle=":", label="学習履歴終点：59,990")
    ax.set_yscale("log")
    ax.set_xlabel("学習ステップ")
    ax.set_ylabel("教示動作への適合誤差（対数軸）")
    ax.set_title("本番運用モデルの学習損失は低下したが、検証曲線は未記録", loc="left")
    ax.legend(frameon=False, ncol=2)
    ax.grid(which="both", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.985,
        0.93,
        "初期値 0.0677\n運用モデルは学習中盤\n最終値 0.000671",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=.5", fc=LIGHT, ec="none"),
    )
    save(fig, "baseline_training_loss")


def figure_experiment_funnel() -> None:
    inventory = load_json("wandb_experiment_inventory.json")["family_summary"]
    stages = ["実行試行", "学習履歴あり", "1万ステップ到達", "2.5万ステップ到達", "プロセス終了記録"]
    keys = ["run_attempts", "runs_with_any_history", "runs_reaching_10000_steps",
            "runs_reaching_25000_steps"]
    families = [
        ("ボトル分別基礎モデル", inventory["baseline_bottle_sorting"], BLUE),
        ("洗浄・挿入の探索", inventory["rinse_or_insertion_exploration"], TEAL),
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
    ax.set_xlabel("実行記録数")
    ax.set_xlim(0, 25)
    ax.set_title("学習試行のファネル：41回の工学的試行を段階別に整理", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "experiment_funnel")


def figure_condition_coverage() -> None:
    cov = load_json("dataset_condition_coverage.json")["category_summary"]
    order = [
        ("向きの変化", "direction_named"),
        ("キャップなし", "no_cap_named"),
        ("初期位置へ復帰", "return_home_named"),
        ("水入りボトル", "water_named"),
        ("ボトル反転", "turn_over_named"),
        ("自由回転キャップ", "free_spinning_named"),
    ]
    labels = [x[0] for x in order]
    episodes = [cov[x[1]]["episodes"] for x in order]
    weighted = [cov[x[1]]["deployed_weighted_episodes"] for x in order]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.barh(y, weighted, color="#DCECF4", height=.62, label="運用モデル学習での等価サンプリング量")
    ax.barh(y, episodes, color=BLUE, height=.36, label="実在する一意な軌跡")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("軌跡数／等価サンプリング数")
    ax.set_title("学習データは複数条件を意図的に含むが、分布は不均衡", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yy, (real, exp) in enumerate(zip(episodes, weighted)):
        ax.text(real + 5, yy, f"{real}", va="center", color=NAVY, fontsize=8.5)
        if exp > real:
            ax.text(exp + 5, yy, f"等価 {exp}", va="center", color=AMBER, fontsize=8.5)
    save(fig, "condition_coverage")


def figure_prompt_mismatch() -> None:
    prompt = load_json("dataset_condition_coverage.json")["prompt_summary"]
    cross = load_json("dataset_condition_coverage.json")["no_cap_prompt_cross_check"]
    labels = ["無条件の「開栓」短文指示", "長文だが無条件で開栓", "「キャップ有り時のみ」と明示"]
    values = [
        prompt["short_unconditional_unscrew"]["episodes"],
        prompt["long_but_unconditional_cap_step"]["episodes"],
        prompt["conditional_on_cap_presence"]["episodes"],
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.7), gridspec_kw={"width_ratios": [1.4, 1]})
    colors = [BLUE, AMBER, GREEN]
    bars = ax1.barh(labels[::-1], values[::-1], color=colors[::-1], height=.56)
    for b, v in zip(bars, values[::-1]):
        ax1.text(v + 18, b.get_y() + b.get_height() / 2, f"{v:,} 軌跡", va="center", weight="bold")
    ax1.set_xlim(0, 1050)
    ax1.set_xlabel("軌跡数")
    ax1.set_title("タスク指示文における条件の網羅", loc="left")
    ax1.grid(axis="x")
    ax1.spines[["top", "right", "left"]].set_visible(False)

    ax2.axis("off")
    rounded_box(ax2, (.08, .55), .84, .27, "キャップなしデータの交差確認",
                f"{cross['repositories']}データ保管単位／{cross['episodes']}軌跡", RED, size=11)
    rounded_box(ax2, (.08, .15), .84, .27, "「キャップなしなら開栓を省略」と明示",
                "0データ保管単位", RED, size=11)
    ax2.annotate("", xy=(.5, .43), xytext=(.5, .55),
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax2.text(.5, .06, "現場の「キャップなしでも空回し」と整合するが、\n因果確認には対条件試験が必要。",
             ha="center", va="center", fontsize=9, color=GRAY)
    save(fig, "prompt_condition_gap")


def figure_attention_share() -> None:
    df = pd.read_csv(DATA / "attention_camera_share.csv")
    first_mask = df.groupby("run_id").cumcount() == 0
    df = df[~first_mask].copy()
    columns = ["cam_high_share", "cam_left_wrist_share", "cam_right_wrist_share"]
    labels = ["上部俯瞰", "左手首近景", "右手首近景"]
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
    ax.set_ylabel("3視野における注意割合（%）")
    ax.set_title("8,223件の実記録：操作点に近い手首視野をより多く参照", loc="left")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    medians = [(df[c] * 100).median() for c in columns]
    for x, med in enumerate(medians, start=1):
        ax.text(x, med + 1.2, f"中央値 {med:.1f}%", ha="center", fontsize=8.5, weight="bold")
    save(fig, "attention_camera_share")


def figure_rlt_eval() -> None:
    df = pd.read_csv(DATA / "round_eval.csv")
    last = df.sort_values(["round", "step"]).groupby("round", as_index=False).tail(1)
    x = last["round"].to_numpy()
    mae = last["val_actor_mae"].to_numpy()
    critic = last["val_critic_loss"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(9.6, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(x, mae, marker="o", linewidth=2.2, color=BLUE, label="検証動作誤差")
    ax2.plot(x, critic, marker="s", linewidth=1.8, color=AMBER, label="価値推定誤差")
    ax1.set_xlabel("連続研究ラウンド")
    ax1.set_ylabel("検証動作誤差（低いほど良い）", color=BLUE)
    ax2.set_ylabel("価値推定誤差（低いほど良い）", color=AMBER)
    ax1.set_xticks(x)
    ax1.grid(axis="y")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.set_title("強化学習は学習可能な閉ループを形成したが、証拠はオフライン検証段階", loc="left")
    improvement = (mae[0] - mae[-1]) / mae[0] * 100
    ax1.text(0.02, .08, f"動作誤差 {mae[0]:.3f} → {mae[-1]:.3f}\n相対低下 約{improvement:.1f}%",
             transform=ax1.transAxes, color=BLUE, fontsize=9,
             bbox=dict(boxstyle="round,pad=.45", fc=LIGHT, ec="none"))
    ax1.text(.98, .08, "同条件の基礎モデル比較なし\n実機成功率の記録なし",
             transform=ax1.transAxes, ha="right", color=RED, fontsize=9,
             bbox=dict(boxstyle="round,pad=.45", fc="#FFF2F0", ec="none"))
    save(fig, "rlt_offline_validation")


def figure_evidence_grade() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (.03, .29), .27, .48, "区分A｜再検証可能な事実",
                "実機運用モデル\n学習履歴と設定\n1,051学習軌跡\n8,223アテンション記録", GREEN, size=11)
    rounded_box(ax, (.365, .29), .27, .48, "区分B｜現場観測による概算",
                "完全な分別サイクル\n連続稼働 >1時間\n平均 約2本/分\n空回し・逆向き把持", AMBER, size=11)
    rounded_box(ax, (.70, .29), .27, .48, "区分C｜今後の標準測定対象",
                "タスク成功率\n条件別成功率\n失敗類型の頻度\n処理量の信頼区間", RED, size=11)
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
        (.025, .56, .15, .25, "3視野", "上部俯瞰\n左手首近景\n右手首近景", BLUE),
        (.025, .17, .15, .25, "ロボット状態", "両腕関節\n両グリッパ位置\n計14変数", TEAL),
        (.23, .36, .18, .28, "統合理解", "画像・動作状態・\nタスク指示を\n一つの表現へ", NAVY),
        (.47, .36, .18, .28, "動作生成", "ノイズで初期化した\n候補を10回更新し\n将来動作を生成", AMBER),
        (.71, .36, .15, .28, "動作列", "将来50制御時刻を\n一括予測\n各時刻14変数", BLUE),
        (.86 + TERMINAL_BOX_GAP, .36, .065, .28, "両腕", "50 Hz\n連続実行", GREEN),
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
    ax.text(.55, .77, "学習：予測した「修正方向」を教示動作の方向へ近づける",
            ha="center", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=.45", fc=LIGHT, ec="none"))
    ax.set_title("本番運用モデル：3視野から両腕の将来動作列を連続生成", loc="left", pad=12)
    save(fig, "model_dataflow")


def figure_task_timeline() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1", "検出・接近", "複数ボトルのランダム配置"),
        ("2", "左手で把持", "安定把持して持ち上げ"),
        ("3", "右手で蓋探索", "両腕を接触点へ"),
        ("4", "協調して開栓", "左で保持、右で回転"),
        ("5", "分離・確認", "キャップの離脱"),
        ("6", "分別投入", "本体と蓋を別容器へ"),
        ("7", "次周期へ復帰", "長時間反復"),
    ]
    xs = np.linspace(.07, .93, len(stages))
    ax.plot(xs, [.54] * len(xs), color="#C9D3DC", linewidth=3, zorder=0)
    for i, (x, (num, title, body)) in enumerate(zip(xs, stages)):
        color = [BLUE, TEAL, AMBER, RED, AMBER, GREEN, NAVY][i]
        ax.scatter([x], [.54], s=850, color=color, edgecolor="white", linewidth=2, zorder=2)
        ax.text(x, .54, num, ha="center", va="center", color="white", weight="bold", fontsize=12)
        ax.text(x, .35 if i % 2 == 0 else .76, title, ha="center", weight="bold", color=color, fontsize=9)
        ax.text(x, .26 if i % 2 == 0 else .85, body, ha="center", color=GRAY, fontsize=8)
    ax.set_title("1本の処理は単発把持ではなく、7段階の双腕協調タスク", loc="left")
    save(fig, "task_timeline")


def figure_software_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (.03, .40, .16, .30, "専用収集ソフト", "連続収集\nペダル／キー操作\n破棄・再収録\n安全確認", BLUE),
        (.235, .40, .16, .30, "データ編集センター", "51有効プロジェクト\n閲覧・ラベル付け\n洗浄・生成\n進捗管理", TEAL),
        (.44, .40, .16, .30, "公開・バージョン管理", "多視点映像\n軌跡と指示文\n確定版\n追跡可能な出所", NAVY),
        (.645, .40, .16, .30, "モデル学習", "方式試行\n品質確認\n重点条件抽出\nモデル保存", AMBER),
        (.85, .40, .12, .30, "現場運用", "3視野\n双腕動作\n連続再計画", GREEN),
    ]
    for x, y, w, h, title, body, color in items:
        rounded_box(ax, (x, y), w, h, title, body, color, size=10)
    for i in range(len(items) - 1):
        x0 = items[i][0] + items[i][2]
        x1 = items[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0, .55), (x1, .55), arrowstyle="-|>",
                                     mutation_scale=14, color=GRAY, lw=1.8))
    ax.text(.315, .23, "データ修正・再収録", ha="center", fontsize=9, color=RED)
    ax.add_patch(FancyArrowPatch((.44, .37), (.19, .37), arrowstyle="-|>",
                                 mutation_scale=13, color=RED, lw=1.3,
                                 connectionstyle="arc3,rad=-.25"))
    ax.text(.745, .23, "失敗条件のフィードバック", ha="center", fontsize=9, color=RED)
    ax.add_patch(FancyArrowPatch((.89, .36), (.60, .36), arrowstyle="-|>",
                                 mutation_scale=13, color=RED, lw=1.3,
                                 connectionstyle="arc3,rad=-.23"))
    ax.set_title("初年度に構築したのはモデル単体ではなく、再利用可能なデータ・学習・運用閉ループ", loc="left")
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
        ("データ資産", f"{platform['projects']}プロジェクト", f"{platform['declared_episodes']:,}軌跡｜約16.15時間", NAVY),
        ("本番学習データ", f"{deployed['unique_episodes']:,}軌跡", f"{deployed['unique_frames']:,}フレーム｜25確定版", BLUE),
        ("学習探索", f"{runs['run_attempts']}回試行", f"{runs['unique_config_names']}設定｜6種類のバッチ", TEAL),
        ("本番学習", "59,990ステップ", "6,059記録｜約51.4時間", AMBER),
        ("運用モデル", "8.38億パラメータ", f"ステップ{checkpoint['directory_step']:,}｜3視野", GREEN),
        ("視覚監査", f"{attention['total_samples']:,}サンプル", f"{attention['manifest_count']}実行記録一覧｜3視野", CYAN),
        ("強化学習研究", f"{rlt['raw_dataset']['episodes']}軌跡", f"{rlt['training_replay']['transitions']['sum']:,}学習区間｜5ラウンド", RED),
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
    ax.set_title("データ生産から現場運用まで：初年度に形成した7種の検証可能な工学資産",
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
        ("P0", "結果の信頼性確立", "固定試験条件｜1本ごとの計数｜自動成否判定｜失敗類型の記録", RED),
        ("P1", "基礎能力の強化", "「キャップなしは省略」とボトル口の向きを追加｜対照評価｜把持・開栓安定化", AMBER),
        ("P2", "強化学習で最適化", "同条件の基礎モデル比較｜オフライン選別｜小規模で安全な実機検証", BLUE),
        ("P3", "管挿入洗浄へ展開", "デジタルツイン校正｜ミリメートル級計測｜視覚・接触協調｜段階別評価", TEAL),
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
    ax.text(.99, .93, "まず測る → データ補強 → 最適化 → 高精度挿入へ",
            ha="right", fontsize=10, color=GRAY)
    ax.set_title("次年度ロードマップ：各工程は現在のエビデンス不足に対応", loc="left")
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

    label("左作業アーム", (340, 430, 590, 505), (520, 700), (36, 116, 181))
    label("右作業アーム", (1110, 410, 1375, 485), (1190, 700), (22, 139, 131))
    label("上部俯瞰カメラ", (690, 330, 1010, 405), TOP_CAMERA_TARGET, (230, 159, 53))
    label("操作者側の教示アーム", (65, 920, 330, 995), (285, 850), (102, 112, 133))
    label("操作者側の教示アーム", (1370, 900, 1635, 975), (1440, 830), (102, 112, 133))
    draw.rounded_rectangle((475, 750, 1230, 930), radius=25, outline=(198, 83, 76, 230),
                           width=6, fill=(198, 83, 76, 28))
    draw.rounded_rectangle((650, 825, 1050, 900), radius=16, fill=(255, 255, 255, 220))
    draw.text((850, 862), "ロボット実作業領域", font=bold, fill=(198, 83, 76, 255),
              anchor="mm")
    draw.rounded_rectangle((25, 25, 670, 100), radius=15, fill=(20, 43, 74, 220))
    draw.text((48, 62), "実機設備写真｜静止・休止状態（タスク実行中ではない）",
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
        ("通常の全工程教示", "ordinary_full_task__episode_0009"),
        ("複数方向のボトル", "direction__episode_0052"),
        ("キャップなし条件", "no_cap__episode_0006"),
        ("ボトル反転条件", "turn_over__episode_0022"),
    ]
    fig, axes = plt.subplots(len(categories), 3, figsize=(10.6, 9.1))
    for row, (title, prefix) in enumerate(categories):
        for col, pct in enumerate([20, 50, 80]):
            path = BUILD / "hf_training_keyframes" / f"{prefix}__p{pct}.png"
            axes[row, col].imshow(Image.open(path))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(["前半", "中盤", "後半"][col], fontsize=10, weight="bold")
        axes[row, 0].text(-.03, .5, title, transform=axes[row, 0].transAxes,
                          ha="right", va="center", rotation=90, fontsize=10,
                          weight="bold", color=NAVY)
    fig.suptitle("実学習教示の代表フレーム：場面の多様性を示すが、自律試験結果ではない",
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
    for i, (name, share) in enumerate(zip(["上部俯瞰", "左手首近景", "右手首近景"], shares)):
        ax.text((i + .5) * segment, 18, f"{name}｜本例の割合 {share:.1f}%",
                ha="center", va="top", color="white", fontsize=9, weight="bold",
                bbox=dict(boxstyle="round,pad=.35", fc=(0.05, .12, .20, .78), ec="none"))
    fig.suptitle("実行時のアテンションマップ例：色付き領域は動作生成時に重点参照した位置",
                 fontsize=13, weight="bold", x=.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(FIG / "attention_real_example.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_plot_manifest() -> None:
    records = [
        ("data_scope", "データ資産の範囲", "全資産と本番学習部分集合の違い"),
        ("sampling_exposure", "重点サンプリング", "一意な軌跡と等価サンプリング量の違い"),
        ("baseline_episode_length_distribution", "教示時間", "軌跡長の分布"),
        ("baseline_training_loss", "学習の実在性", "学習誤差は低下したが検証成功率はない"),
        ("experiment_funnel", "工学的作業量", "実行試行と十分に進んだ学習の違い"),
        ("condition_coverage", "条件網羅", "方向、キャップなし、反転などの分布"),
        ("prompt_condition_gap", "空回しリスク", "キャップなしにも無条件開栓指示を使用"),
        ("attention_camera_share", "視覚利用", "3視野の注意割合"),
        ("attention_real_example", "アテンションマップ", "実記録の熱領域と解釈限界"),
        ("rlt_offline_validation", "強化学習探索", "オフライン学習閉ループのみを示す"),
        ("evidence_grade", "結論の確度", "確認事実、現場概算、未測定を区別"),
        ("model_dataflow", "手法", "観測から双腕動作までの流れ"),
        ("task_timeline", "タスク難度", "7段階の協調動作と誤差蓄積"),
        ("software_data_pipeline", "基盤作業量", "収集、編集、公開、学習、運用の閉ループ"),
        ("engineering_workload_dashboard", "工学的作業強度", "7種の検証可能な工学資産"),
        ("next_year_roadmap", "次年度", "優先順位と依存関係"),
        ("training_demonstration_keyframes", "実学習画像", "データ網羅を示し、自律成功を示さない"),
        ("aloha_formal_photo_annotated", "実機設備", "静止状態の設備構成"),
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
