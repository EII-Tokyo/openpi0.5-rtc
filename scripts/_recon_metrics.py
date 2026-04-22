
import io
import contextlib
import numpy as np
from transformers import AutoProcessor
from openpi.shared import normalize as _normalize
from scripts.train_fast_tokenizer import _load_from_lerobot_preprocessed

REPO = "lyl472324464/twist_subset_balanced_100k_448_multi_repo_300mb"
NORM = Path("output/fast_tokenizers/twist_subset_balanced_100k_448_multi_repo_300mb/norm_stats")
BASE = Path("output/fast_tokenizers/twist_subset_balanced_100k_448_multi_repo_300mb")
PATHS = {
    "scale10": BASE / "fast_tokenizer_v2048_100k",
    "scale1": BASE / "fast_tokenizer_v2048_100k_scale1",
}

def metrics(chunks, proc, name, n_eval):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(chunks), size=min(n_eval, len(chunks)), replace=False)
    buf = io.StringIO()
    mses = []
    fail = 0
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for j in idx:
            c = chunks[j]
            toks = proc(c[None, ...])
            recon = proc.decode(toks, time_horizon=50, action_dim=14)[0]
            d = recon.astype(np.float64) - c.astype(np.float64)
            mses.append(float(np.mean(d * d)))
            if float(np.mean((recon.astype(np.float64)) ** 2)) < 1e-12 and float(np.mean((c.astype(np.float64)) ** 2)) > 1e-6:
                fail += 1
    mses = np.array(mses, dtype=np.float64)
    flat_orig = chunks[idx].reshape(-1).astype(np.float64)
    var = float(np.var(flat_orig))
    mean_mse = float(mses.mean())
    rmse = float(np.sqrt(mean_mse))
    r2 = 1.0 - mean_mse / var if var > 1e-12 else float("nan")
    rel_rmse = rmse / float(np.std(flat_orig))
    snr_db = 10.0 * np.log10(var / mean_mse) if mean_mse > 1e-20 else float("inf")
    return dict(name=name, n=len(idx), mean_mse=mean_mse, rmse=rmse, r2=r2, rel_rmse_std=rel_rmse, snr_db_versus_var=float(snr_db), decode_suspicious_fail=fail)

norm = _normalize.load(NORM)
chunks = _load_from_lerobot_preprocessed(REPO, action_key="action", state_key="observation.state", chunk_len=50, stride=1, adapt_to_pi=True, use_quantiles=True, norm_stats=norm)
N = 20000
lines = ["n_chunks_total=%d eval_subsample=%d" % (len(chunks), N)]
for k, path in PATHS.items():
    proc = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    m = metrics(chunks, proc, k, N)
    lines.append(str(m))
out = Path("output/fast_tokenizers/twist_subset_balanced_100k_448_multi_repo_300mb/reconstruction_metrics_20k.jsonl")
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
