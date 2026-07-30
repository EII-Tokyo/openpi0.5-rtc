#!/usr/bin/env python3
"""Audit every figure referenced by the public ALOHA report."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "sections"
FIGURES = ROOT / "figures"
ARTIFACT = ROOT / "artifacts" / "figure_audit.json"
REVIEW = ROOT / "audit" / "figure_review.md"
GENERATOR = ROOT / "scripts" / "generate_annual_report_figures.py"

CONTRACTS = {
    "cover_concept.png": (
        "报告面向的完整双臂瓶子分拣任务是什么？",
        "封面概括取瓶、旋盖和分类投放的完整任务链。",
        "概念示意图不是实验照片。",
        "artifacts/cover_concept_provenance.json",
    ),
    "evidence_grade.pdf": (
        "当前成果分别具有怎样的证据强度？",
        "可复核事实、现场观察和待量化指标需要分层陈述。",
        "证据等级不是任务成功率。",
        "artifacts/field_observation_claims.json",
    ),
    "aloha_formal_photo_annotated.png": (
        "正式 ALOHA 工作站由哪些主要部分组成？",
        "正式设备包含示教臂、工作臂、相机和任务工作区。",
        "照片拍摄时机器人处于静止休眠状态。",
        "artifacts/official_photo_provenance.json",
    ),
    "task_timeline.pdf": (
        "单个瓶子的完整处理链包含哪些阶段？",
        "任务包含七个相互依赖的长流程阶段。",
        "阶段划分不能代替逐阶段成功率。",
        "artifacts/plot_manifest.json",
    ),
    "software_data_pipeline.pdf": (
        "第一年度形成了怎样的数据到部署闭环？",
        "工作覆盖采集、编辑、训练、部署、观察和问题回流。",
        "当前软件能力不代表历史数据都使用了全部后来功能。",
        "artifacts/plot_manifest.json",
    ),
    "data_scope.pdf": (
        "数据总资产、正式训练子集和过滤后输入分别有多大？",
        "三层数据口径必须分开统计。",
        "总资产不能写成正式模型训练量。",
        "artifacts/dataset_statistics.json",
    ),
    "sampling_exposure.pdf": (
        "困难样本如何在训练中获得更多出现机会？",
        "重复采样提高重点场景的训练暴露。",
        "等效采样数量不是新增独立轨迹。",
        "artifacts/dataset_statistics.json",
    ),
    "condition_coverage.pdf": (
        "正式训练数据覆盖了哪些困难条件？",
        "数据包含方向、无盖、翻转和自由旋转等条件。",
        "名称分类允许重叠且不表示各条件成功率。",
        "artifacts/plot_data/dataset_condition_coverage.csv",
    ),
    "baseline_episode_length_distribution.pdf": (
        "正式训练轨迹的长度分布如何？",
        "1,051 条轨迹覆盖短、中、长多种示教时长。",
        "极长尾只为绘图可读性截到 99% 分位。",
        "artifacts/plot_data/baseline_training_episode_lengths.csv",
    ),
    "training_demonstration_keyframes.png": (
        "真实训练数据中的典型画面是什么样？",
        "固定关键帧展示多种真实示教条件。",
        "示教关键帧不是自主测试结果。",
        "artifacts/hf_training_keyframe_manifest.json",
    ),
    "model_dataflow.pdf": (
        "三路视觉与机器人状态如何形成连续双臂动作？",
        "正式部署模型输出未来 50 个控制时刻的双臂动作。",
        "示意图只描述正式部署模型，不代表其他候选模型。",
        "artifacts/checkpoint_metadata.json",
    ),
    "attention_real_example.png": (
        "一次真实运行中三路视野的关注位置是什么样？",
        "样例展示顶部和双腕视野的真实注意力热区。",
        "记录没有成功或失败标签。",
        "artifacts/attention_audit.json",
    ),
    "attention_camera_share.pdf": (
        "8,223 个样本中三路视野的注意力份额如何分布？",
        "腕部近景整体高于顶部总览。",
        "注意力份额不是失败原因的因果解释。",
        "artifacts/plot_data/attention_camera_share.csv",
    ),
    "experiment_funnel.pdf": (
        "41 次训练尝试中有多少留下历史并运行到较远阶段？",
        "训练谱系体现工程迭代量和运行稳定性。",
        "运行状态不等同于模型真机任务成功。",
        "artifacts/plot_data/wandb_experiment_inventory.csv",
    ),
    "baseline_training_loss.pdf": (
        "正式模型是否持续拟合示教动作？",
        "完整训练历史显示动作拟合误差持续下降。",
        "训练误差下降不能代替真机成功率。",
        "artifacts/plot_data/baseline_training_history.csv",
    ),
    "prompt_condition_gap.pdf": (
        "无盖数据的任务指令是否明确要求跳过旋拧？",
        "无盖条件仍普遍使用无条件旋盖指令。",
        "指令缺口是风险证据，不是唯一原因证明。",
        "artifacts/dataset_condition_coverage.json",
    ),
    "rlt_offline_validation.pdf": (
        "强化学习研究闭环目前达到什么程度？",
        "连续轮次形成了可比较的离线动作与价值指标。",
        "离线改善不能写成真机能力提升。",
        "artifacts/plot_data/round_eval.csv",
    ),
    "next_year_roadmap.pdf": (
        "下一年度工作为什么按当前优先级推进？",
        "路线从可信评估到基础能力、强化学习和高精度插入逐级推进。",
        "插管目标尚未完成，毫米级要求仍需实测标定。",
        "artifacts/plot_manifest.json",
    ),
}


def referenced_figures() -> set[str]:
    sources = [ROOT / "aloha_bottle_cap_report.tex", *sorted(SECTIONS.glob("*.tex"))]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    direct = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    macro = re.findall(r"\\figfull(?:\[[^\]]*\])?\{([^}]+)\}", text)
    return {
        Path(value).name
        for value in direct + macro
        if not value.startswith("#")
    }


def main() -> None:
    violations: list[str] = []
    generator_source = GENERATOR.read_text(encoding="utf-8")
    for phrase in ("这张图回答：", "｜边界：", "“标记结束”只代表进程状态"):
        if phrase in generator_source:
            violations.append(f"plot-canvas prose remains in generator: {phrase}")

    referenced = referenced_figures()
    missing_contracts = sorted(referenced - CONTRACTS.keys())
    extra_contracts = sorted(CONTRACTS.keys() - referenced)
    for name in missing_contracts:
        violations.append(f"missing figure contract: {name}")
    for name in extra_contracts:
        violations.append(f"contract is not referenced by report: {name}")

    questions: dict[str, str] = {}
    entries = []
    for name in sorted(referenced & CONTRACTS.keys()):
        question, claim, boundary, source = CONTRACTS[name]
        if question in questions:
            violations.append(
                f"duplicate figure question: {name} and {questions[question]}: {question}"
            )
        questions[question] = name
        path = FIGURES / name
        if not path.is_file():
            violations.append(f"missing figure file: {name}")
        source_path = ROOT / source
        if not source_path.is_file():
            violations.append(f"missing source artifact: {name}: {source}")
        dimensions = None
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file():
            with Image.open(path) as image:
                dimensions = [image.width, image.height]
            if image.width < 1200:
                violations.append(f"raster figure width below 1200 px: {name}: {image.width}")
        entries.append(
            {
                "figure": f"figures/{name}",
                "question": question,
                "claim": claim,
                "evidence_boundary": boundary,
                "source_artifact": source,
                "dimensions_px": dimensions,
                "visual_status": "PENDING" if violations else "PASS",
            }
        )

    status = "PASS" if not violations else "FAIL"
    if status == "PASS":
        for entry in entries:
            entry["visual_status"] = "PASS"
    payload = {
        "status": status,
        "referenced_figure_count": len(referenced),
        "entries": entries,
        "violations": violations,
    }
    ARTIFACT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = [
        "# 逐图视觉与证据审查",
        "",
        f"- 状态：**{status}**",
        f"- 正文实际引用图：{len(referenced)}",
        f"- 违规项：{len(violations)}",
        "",
        "| 图文件 | 唯一问题 | 支撑结论 | 证据边界 | 状态 |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        rows.append(
            "| {figure} | {question} | {claim} | {boundary} | {status} |".format(
                figure=entry["figure"],
                question=entry["question"],
                claim=entry["claim"],
                boundary=entry["evidence_boundary"],
                status=entry["visual_status"],
            )
        )
    if violations:
        rows.extend(["", "## 违规项", "", *[f"- {item}" for item in violations]])
    REVIEW.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "referenced_figures": len(referenced),
                "violations": len(violations),
            },
            ensure_ascii=False,
        )
    )
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
