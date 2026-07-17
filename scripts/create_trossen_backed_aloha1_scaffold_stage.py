from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TROSSEN_USD = REPO_ROOT / "external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"
DEFAULT_CONTRACT_JSON = (
    REPO_ROOT
    / "reports/aloha1_isaac_adaptation/phase3_scaffold_contract_20260717/trossen_backed_aloha1_scaffold_contract.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "local_eval_assets/aloha1_trossen_backed_scaffold_20260717"


def _relative_reference(from_file: Path, target: Path) -> str:
    return Path(shutil.os.path.relpath(target.resolve(), from_file.resolve().parent)).as_posix()


def _write_usda(output_usd: Path, trossen_usd: Path) -> None:
    reference = _relative_reference(output_usd, trossen_usd)
    text = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "Aloha1TrossenBackedScaffold" (
        prepend references = @{reference}@
    )
    {{
        string adaptation_note = "Trossen stationary_ai runtime scaffold for ALOHA1. Physical/electrical ALOHA1 semantics are not confirmed by this layer."
        string adapter_contract = "trossen_backed_aloha1_scaffold_contract.json"
        string source_trossen_usd = "{reference}"
    }}
}}
'''
    output_usd.write_text(text, encoding="utf-8")


def _write_readme(output_dir: Path, output_usd: Path, trossen_usd: Path, contract_path: Path) -> None:
    readme = f"""# Trossen-Backed ALOHA1 Scaffold

This is a scaffold stage for adapting Trossen `stationary_ai` to the user's ALOHA1.

It deliberately uses Trossen as the Isaac runtime structure standard:

- articulation structure;
- meshes;
- colliders;
- cameras;
- materials.

It does not claim Trossen physical or electrical semantics are correct for ALOHA1.
Those fields remain governed by the adapter contract.

## Files

- USD: `{output_usd.name}`
- Adapter contract: `{contract_path.name}`
- Source Trossen USD: `{trossen_usd}`

## Gate

This scaffold is acceptable only if a headless Isaac runtime report shows:

- one bimanual articulation initializes;
- mesh, collider, camera, and material counts are nonzero;
- unresolved robot visual/collision references are absent;
- ALOHA1 adapter fields remain marked as requiring verification until proven.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Trossen-backed ALOHA1 scaffold USD layer.")
    parser.add_argument("--trossen-usd", type=Path, default=DEFAULT_TROSSEN_USD)
    parser.add_argument("--contract-json", type=Path, default=DEFAULT_CONTRACT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.trossen_usd.exists():
        raise FileNotFoundError(args.trossen_usd)
    if not args.contract_json.exists():
        raise FileNotFoundError(args.contract_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_usd = args.output_dir / "aloha1_trossen_backed_scaffold.usda"
    output_contract = args.output_dir / "trossen_backed_aloha1_scaffold_contract.json"
    contract = json.loads(args.contract_json.read_text(encoding="utf-8"))
    output_contract.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_usda(output_usd, args.trossen_usd)
    _write_readme(args.output_dir, output_usd, args.trossen_usd, output_contract)
    print(json.dumps({"usd": str(output_usd), "contract": str(output_contract), "source_trossen_usd": str(args.trossen_usd)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
