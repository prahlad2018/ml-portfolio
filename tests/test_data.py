from src.ml_portfolio.data.loaders import load_sample

def test_load_sample():
    df = load_sample("data")
    assert not df.empty
    assert "y" in df.columns
