from src.harness.metrics import (
    efficiency,
    f_eff_at_cluster,
    fer_with_ci,
    h2,
    optimal_cluster_size,
    predicted_secret_key_rate,
    wilson_ci,
)
from src.harness.plots import make_slide_deck, per_qa_summary
from src.harness.runner import (
    AlgorithmRunner,
    generate_frames,
    make_borisov,
    make_cascade,
    make_mueller,
    run_mismatch_sweep,
    run_qber_sweep,
    save_records,
)

__all__ = [
    "AlgorithmRunner",
    "generate_frames",
    "make_borisov",
    "make_cascade",
    "make_mueller",
    "run_mismatch_sweep",
    "run_qber_sweep",
    "save_records",
    # metrics
    "h2",
    "efficiency",
    "f_eff_at_cluster",
    "optimal_cluster_size",
    "wilson_ci",
    "fer_with_ci",
    "predicted_secret_key_rate",
    # plots
    "per_qa_summary",
    "make_slide_deck",
]
