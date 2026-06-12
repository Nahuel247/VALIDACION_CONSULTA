from pathlib import Path

from src.data.loader import load_dataset


def test_load_dataset_reads_expected_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("text;label\nHola;valida\nAdios;no_valida\n", encoding="utf-8")
    df = load_dataset(csv_path)
    assert list(df.columns)[:2] == ["text", "label"]
    assert len(df) == 2
