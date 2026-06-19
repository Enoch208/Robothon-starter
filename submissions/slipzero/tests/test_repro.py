from eval import run, summarize

_CONFIG = "config/default.yaml"


def test_eval_is_deterministic():
    first = run(trials=4, seed=0, config_path=_CONFIG)
    second = run(trials=4, seed=0, config_path=_CONFIG)
    assert first == second


def test_closed_loop_not_worse_than_baseline():
    summary = summarize(run(trials=6, seed=0, config_path=_CONFIG))
    assert summary["success_rate"] >= summary["baseline_success_rate"]
    assert summary["drops"] <= summary["baseline_drops"]
