import json

import numpy as np
import pandas as pd

from analysis.forecasts import live_forecast, load_forecasts
from scripts import backtest as backtest_script
from scripts import benchmark as benchmark_script
from scripts import evaluate as evaluate_script
from scripts import pipeline as pipeline_script
from scripts import predict as predict_script
from scripts import report as report_script
from scripts import train as train_script


def test_train_evaluate_predict_backtest_report(tiny_config, tmp_path):
    predictor, history = train_script.train(tiny_config, device="cpu")
    assert 1 <= len(history) <= tiny_config["num_epochs"]
    assert all(np.isfinite(epoch["val_loss"]) for epoch in history)
    model_path = tiny_config["model_path"]

    plot_path = tmp_path / "forecasts.png"
    results = evaluate_script.evaluate(model_path, device="cpu", plot_path=plot_path)
    assert results["samples"] > 0
    assert np.isfinite(results["intervals"]["90"]["rolling"]["coverage"])
    assert plot_path.stat().st_size > 0
    report = evaluate_script.format_report(results)
    assert "Naive (last close)" in report and "rolling conformal" in report

    predictor, frame, table = load_forecasts(model_path, device="cpu")
    live = live_forecast(table, predictor.quantiles)
    assert live["target_candle"] == frame.index[-1] + pd.Timedelta(hours=1)
    assert "Chance of a higher close" in predict_script.format_forecast(live)
    history_frame = predict_script.forecast_history(table, predictor.quantiles)
    assert set(history_frame["split"]) == {"val", "test", "live"}
    assert (history_frame["lower_90"] < history_frame["upper_90"]).all()

    summary, backtests = backtest_script.backtest(model_path, device="cpu", plot_path=tmp_path / "equity.png")
    assert set(summary["strategies"]) == set(backtests)
    assert "Buy & hold" in backtest_script.format_summary(summary)
    assert (tmp_path / "equity.png").stat().st_size > 0

    dashboard = tmp_path / "site" / "index.html"
    numbers = report_script.build_report(model_path, output=dashboard, device="cpu")
    page = dashboard.read_text()
    for heading in ("Forecast with calibrated intervals", "Risk intervals", "Backtest, net of costs", "What drives"):
        assert heading in page
    assert "cdn.plot.ly" in page
    assert set(numbers) == {"generated", "live", "evaluation", "backtest", "importance"}


def test_command_line_entry_points(tiny_config, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config))
    model_path = str(tmp_path / "cli_model.pt")

    train_script.main(
        ["--config", str(config_path), "--model-path", model_path, "--epochs", "1", "--model-type", "tcn"]
    )

    metrics_path = tmp_path / "metrics.json"
    evaluate_script.main(["--model-path", model_path, "--split", "val", "--metrics-out", str(metrics_path)])
    metrics = json.loads(metrics_path.read_text())
    assert metrics["split"] == "val" and metrics["model_type"] == "tcn"

    predict_script.main(["--model-path", model_path, "--history-out", str(tmp_path / "history.csv")])
    backtest_script.main(["--model-path", model_path, "--fee", "0", "--metrics-out", str(tmp_path / "backtest.json")])
    report_script.main(["--model-path", model_path, "--output", str(tmp_path / "report.html"), "--no-importance"])
    output = capsys.readouterr().out
    assert "Brier score" in output and "Forecast for the" in output and "Volatility-targeted hold" in output
    assert json.loads((tmp_path / "backtest.json").read_text())["fee_rate"] == 0.0
    assert (tmp_path / "history.csv").exists() and (tmp_path / "report.html").exists()


def test_benchmark_ranks_models_and_baselines(tiny_config, tmp_path):
    rows = benchmark_script.run_benchmark({**tiny_config, "num_epochs": 1}, ["linear", "transformer"], "cpu", tmp_path)
    assert [row["name"] for row in rows] == ["Linear", "Transformer", "Historical simulation", "EWMA volatility"]
    table = benchmark_script.format_leaderboard(rows, tiny_config["quantiles"])
    assert table.count("\n") == len(rows) + 1
    assert "EWMA volatility" in table


def test_pipeline_reuses_a_trained_model(tiny_config, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config))
    output_dir = tmp_path / "site"
    args = ["--config", str(config_path), "--skip-fetch", "--reuse-model", "--output-dir", str(output_dir)]

    pipeline_script.main(args)
    first_checkpoint = (tmp_path / "model.pt").stat().st_mtime_ns
    monkeypatch.setattr(pipeline_script, "train", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    pipeline_script.main(args)

    assert (tmp_path / "model.pt").stat().st_mtime_ns == first_checkpoint
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["live"]["predicted_close"] > 0
    assert (output_dir / "index.html").exists()
