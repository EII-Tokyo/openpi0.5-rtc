from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _latest_run(root: Path) -> Path:
    runs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "manifest.jsonl").is_file())
    if not runs:
        raise FileNotFoundError(f"No attention runs found below {root}")
    return runs[-1]


def _read_manifest(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_report(run_dir: Path, output: Path) -> None:
    samples = _read_manifest(run_dir)
    camera_names = samples[0]["camera_order"] if samples else []
    rows = []
    chart_data = []
    for index, sample in enumerate(samples):
        masses = sample["mean_attention_mass"]
        total = sum(masses.values()) or 1.0
        shares = {name: 100.0 * masses[name] / total for name in camera_names}
        chart_data.append({"index": index, "sample": sample["sample"], **shares})
        share_text = " · ".join(f"{html.escape(name)} {shares[name]:.1f}%" for name in camera_names)
        sample_name = html.escape(sample["sample"])
        rows.append(
            f"""
            <button class="sample" data-index="{index}" type="button">
              <img src="{sample_name}/overview.jpg" alt="{sample_name} 三路 attention 叠加图">
              <span><strong>{sample_name}</strong><br>{share_text}<br>
              attention probe {sample["capture_ms"]:.0f} ms</span>
            </button>
            """
        )

    payload = json.dumps({"cameras": camera_names, "samples": chart_data}, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenPI 三路相机 Attention</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
    body {{ max-width: 1500px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 6px; }}
    .note {{ color: #aaa; margin: 0 0 18px; }}
    canvas {{ width: 100%; height: 220px; background: #191919; border: 1px solid #333; border-radius: 8px; }}
    .legend {{ display: flex; gap: 18px; margin: 10px 0 20px; font-size: 13px; }}
    .swatch {{ display: inline-block; width: 12px; height: 3px; margin-right: 6px; vertical-align: middle; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }}
    .sample {{ color: inherit; background: #191919; border: 1px solid #333; border-radius: 8px;
      padding: 8px; text-align: left; cursor: pointer; }}
    .sample:hover, .sample.selected {{ border-color: #eee; }}
    .sample img {{ width: 100%; display: block; border-radius: 5px; margin-bottom: 7px; }}
    .sample span {{ font-size: 13px; line-height: 1.45; }}
    @media (max-width: 560px) {{ .gallery {{ grid-template-columns: 1fr; }} body {{ padding: 10px; }} }}
  </style>
</head>
<body>
  <h1>OpenPI 三路相机 Attention</h1>
  <p class="note">
    动作 token → 视觉 token；折线是每次采样中三路相机分到的 attention 比例。
    缩略图使用最后 6 层、前 10 个动作 token 的平均值。
  </p>
  <canvas id="chart" width="1400" height="220" aria-label="三路相机 attention 比例时间序列"></canvas>
  <div class="legend" id="legend"></div>
  <div class="gallery">{"".join(rows)}</div>
  <script>
    const data = {payload};
    const colors = ["#ffcc66", "#66d9ef", "#f2777a"];
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const pad = {{left: 45, right: 12, top: 14, bottom: 28}};
    const w = canvas.width - pad.left - pad.right;
    const h = canvas.height - pad.top - pad.bottom;
    ctx.strokeStyle = "#444"; ctx.fillStyle = "#aaa"; ctx.font = "12px system-ui";
    for (let p = 0; p <= 100; p += 25) {{
      const y = pad.top + h * (1 - p / 100);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(canvas.width - pad.right, y); ctx.stroke();
      ctx.fillText(`${{p}}%`, 5, y + 4);
    }}
    const denom = Math.max(1, data.samples.length - 1);
    data.cameras.forEach((camera, ci) => {{
      ctx.strokeStyle = colors[ci % colors.length]; ctx.lineWidth = 2; ctx.beginPath();
      data.samples.forEach((sample, i) => {{
        const x = pad.left + w * i / denom;
        const y = pad.top + h * (1 - sample[camera] / 100);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }});
    ctx.fillStyle = "#aaa";
    ctx.fillText("sample 0", pad.left, canvas.height - 7);
    if (data.samples.length) ctx.fillText(`sample ${{data.samples.length - 1}}`, canvas.width - 75, canvas.height - 7);
    document.getElementById("legend").innerHTML = data.cameras.map((camera, i) =>
      `<span><i class="swatch" style="background:${{colors[i % colors.length]}}"></i>${{camera}}</span>`).join("");
    document.querySelectorAll(".sample").forEach(button => button.addEventListener("click", () => {{
      document.querySelectorAll(".sample").forEach(item => item.classList.remove("selected"));
      button.classList.add("selected");
    }}));
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML summary of an OpenPI attention capture run.")
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=Path("attention_debug"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run(args.root)
    output = args.output or run_dir / "report.html"
    render_report(run_dir, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
