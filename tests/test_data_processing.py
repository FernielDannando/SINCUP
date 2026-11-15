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


def test_split_features_target_valid_column():
    from Clusters import split_features_target

    df = pd.DataFrame({
        "age": [20, 30],
        "salary": [12000, 18000],
        "target": [0, 1],
    })

    X, y = split_features_target(df, "target")

    assert list(X.columns) == ["age", "salary"]
    assert y.tolist() == [0, 1]
    assert "target" in df.columns


def test_split_features_target_without_target():
    from Clusters import split_features_target

    df = pd.DataFrame({"age": [20, 30]})

    X, y = split_features_target(df, None)

    pd.testing.assert_frame_equal(X, df)
    assert y is None


def test_split_features_target_missing_column():
    from Clusters import split_features_target

    df = pd.DataFrame({"age": [20, 30]})

    X, y = split_features_target(df, "unknown")

    pd.testing.assert_frame_equal(X, df)
    assert y is None
