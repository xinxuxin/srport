from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .utils import ensure_dir, write_json


def render_eval_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# EPNet Evaluation Report",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Device: `{summary['device']}`",
        f"- Samples: `{summary['samples']}`",
        "",
        "| Dataset | Avg PSNR | Avg SSIM | Samples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, payload in summary["datasets"].items():
        lines.append(
            f"| {name} | {float(payload['average_psnr']):.4f} | "
            f"{float(payload['average_ssim']):.5f} | {int(payload['samples'])} |"
        )
    if "summary" in summary:
        lines.extend(
            [
                "",
                "## Overall",
                "",
                f"- Mean PSNR: `{float(summary['summary']['average_psnr']):.4f}`",
                f"- Mean SSIM: `{float(summary['summary']['average_ssim']):.5f}`",
            ]
        )
    return "\n".join(lines)


def render_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# EPNet Profile Report",
        "",
        f"- Checkpoint: `{profile['checkpoint']}`",
        f"- Device: `{profile['device']}`",
        f"- Parameters: `{int(profile['parameters']):,}`",
        f"- MACs: `{int(profile['macs']):,}`",
        f"- FLOPs: `{int(profile['flops']):,}`",
        f"- Latency: `{float(profile['latency_ms']):.4f} ms`",
        f"- Checkpoint Size: `{int(profile['checkpoint_size_bytes']):,} bytes`",
        f"- Model State Size: `{int(profile['model_state_size_bytes']):,} bytes`",
        f"- Estimated Memory Footprint: `{int(profile['estimated_memory_bytes']):,} bytes`",
        f"- ONNX Export: `{profile.get('onnx_export', 'not-run')}`",
    ]
    return "\n".join(lines)


def write_report_bundle(
    payload: dict[str, Any],
    *,
    output_json: Path | None,
    output_markdown: Path | None,
    renderer: Callable[[dict[str, Any]], str],
) -> None:
    if output_json is not None:
        ensure_dir(output_json.parent)
        write_json(output_json, payload)
    if output_markdown is not None:
        assert output_markdown is not None
        ensure_dir(output_markdown.parent)
        output_markdown.write_text(renderer(payload), encoding="utf-8")
