"""Robot task controls are a capability of the loaded policy, not the web process."""


def task_contract(metadata):
    contract = dict(metadata.get("runtime", {}).get("task_control") or {})
    mode = contract.get("mode", "legacy")
    if mode not in {"legacy", "exp_insert"}:
        raise ValueError(f"Unsupported checkpoint task mode: {mode}")
    if mode == "exp_insert" and contract.get("phase_field") != "task_phase":
        raise ValueError(
            "exp_insert automatic cycles require policy task_phase output; this checkpoint is low-level only. "
            "Connect the trained high-level stage predictor before running automatic bottle cycles."
        )
    return mode
