import pandas as pd

from Clusters import select_numeric


def test_select_numeric_mixed_types():
    df = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [12000.5, 15000.0, 18000.5],
        "name": ["Ana", "Luis", "Pedro"],
        "active": [True, False, True],
    })

    assert select_numeric(df) == [
        "age", "salary", "active"
    ]


def test_select_numeric_without_numeric_columns():
    df = pd.DataFrame({
        "name": ["Ana", "Luis"],
        "city": ["Guadalajara", "Zapopan"],
    })

    assert select_numeric(df) == []
