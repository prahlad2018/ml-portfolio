from src.ml_portfolio.models.registry import get_model

def test_get_model_logreg():
    m = get_model('logreg', max_iter=1000)
    assert hasattr(m, 'fit')
