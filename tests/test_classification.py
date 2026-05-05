import pandas as pd
from ml.classification import train_and_evaluate

def test_cross_validation():
    df = pd.read_csv('data/extended_products.csv')
    # Création d'une cible artificielle pour le test
    df['top_product'] = (df['rating'] > 4.5) & (df['reviews'] > 1000)
    _, metrics = train_and_evaluate(df, model_type='random_forest')
    assert metrics['cv_accuracy_mean'] > 0.5
    assert metrics['cv_f1_mean'] > 0.5