import pandas as pd

from src.credit_risk.config import DATA_PATH
from src.credit_risk.showcase import build_summary


def test_showcase_summary_matches_verified_raw_data():
    summary = build_summary(pd.read_csv(DATA_PATH))

    assert summary["rows"] == 30000
    assert summary["columns"] == 25
    assert summary["missing_cells"] == 0
    assert summary["duplicate_rows"] == 0
    assert summary["duplicate_ids"] == 0
    assert summary["default_count"] == 6636
    assert summary["default_rate"] == 0.2212
    assert summary["target_correlations"][0]["feature"] == "PAY_0"


def test_showcase_creates_an_eda_heatmap(tmp_path):
    from src.credit_risk.showcase import create_eda_charts

    data = pd.read_csv(DATA_PATH)
    summary = build_summary(data)
    create_eda_charts(data, summary, tmp_path)
    assert (tmp_path / "correlation_heatmap.png").exists()
