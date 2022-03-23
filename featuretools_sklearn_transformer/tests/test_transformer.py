import copy

import numpy as np
import pandas as pd
import pytest
from featuretools.demo.mock_customer import load_mock_customer
from featuretools.wrappers import DFSTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def select_numeric(df):
    return df.select_dtypes(exclude=["object", pd.CategoricalDtype])


@pytest.fixture
def es():
    es = load_mock_customer(
        n_customers=15,
        n_products=15,
        n_sessions=75,
        n_transactions=1000,
        random_seed=0,
        return_entityset=True,
    )
    return es


@pytest.fixture
def es_customer_filtered(es):
    new_es = copy.deepcopy(es)
    customers_df = es["customers"]
    sessions_df = es["sessions"]
    products_df = es["products"]
    transactions_df = es["transactions"]
    customer_ids = [1, 2, 3]
    customers_df = customers_df.loc[customer_ids]
    sessions_df = sessions_df[sessions_df["customer_id"].isin(customer_ids)]
    transactions_df = transactions_df[
        transactions_df["session_id"].isin(sessions_df["session_id"].values)
    ]
    products_df = products_df[
        products_df["product_id"].isin(transactions_df["product_id"].values)
    ]
    new_es.replace_dataframe("customers", customers_df)
    new_es.replace_dataframe("sessions", sessions_df)
    new_es.replace_dataframe("transactions", transactions_df)
    new_es.replace_dataframe("products", products_df)

    return new_es


def get_dataframes_and_relationships(es):
    dataframes = {}
    relationships = []

    for df in es.dataframes:
        dataframes[df.ww.name] = (
            df,
            df.ww.index,
            df.ww.time_index,
            df.ww.logical_types,
        )

    for rel in es.relationships:
        relationships.append(
            (
                rel._parent_dataframe_name,
                rel._parent_column_name,
                rel._child_dataframe_name,
                rel._child_column_name,
            )
        )

    return dataframes, relationships


@pytest.fixture
def df(es):
    df = es["customers"].copy()
    df["target"] = np.random.randint(1, 3, df.shape[0])  # 1 or 2 values
    return df


@pytest.fixture
def pipeline():
    pipeline = Pipeline(
        steps=[
            ("ft", DFSTransformer(target_dataframe_name="customers", max_features=20)),
            ("numeric", FunctionTransformer(select_numeric, validate=False)),
            ("imp", SimpleImputer()),
            ("et", ExtraTreesClassifier(n_estimators=10)),
        ]
    )
    return pipeline


def test_sklearn_transformer_with_entityset(es):
    # Using with transformers
    pipeline = Pipeline(
        steps=[
            ("ft", DFSTransformer(target_dataframe_name="customers")),
            ("numeric", FunctionTransformer(select_numeric, validate=False)),
            ("sc", StandardScaler()),
        ]
    )

    X_train = pipeline.fit(es).transform(es)

    assert X_train.shape[0] == 15


def test_sklearn_transformer_with_dataframes_and_relationships(es):
    # Using with transformers
    pipeline = Pipeline(
        steps=[
            ("ft", DFSTransformer(target_dataframe_name="customers")),
            ("numeric", FunctionTransformer(select_numeric, validate=False)),
            ("sc", StandardScaler()),
        ]
    )
    dataframes, relationships = get_dataframes_and_relationships(es)

    X_train = pipeline.fit((dataframes, relationships)).transform(
        (dataframes, relationships)
    )

    assert X_train.shape[0] == 15


def test_sklearn_estimator_with_entityset(df, es, pipeline):
    # Using with estimator
    pipeline.fit(es, y=df.target.values).predict(es)
    result = pipeline.score(es, df.target.values)

    assert isinstance(result, (float))

    # Pickling / Unpickling Pipeline
    # TODO fix this
    # s = pickle.dumps(pipeline)
    # pipe_pickled = pickle.loads(s)
    # result = pipe_pickled.score(df['customer_id'].values, df.target.values)
    # assert isinstance(result, (float))


def test_sklearn_estimator_with_dataframes_and_relationships(df, es, pipeline):
    # Using with estimator
    dataframes, relationships = get_dataframes_and_relationships(es)
    pipeline.fit((dataframes, relationships), y=df.target.values).predict(
        (dataframes, relationships)
    )
    result = pipeline.score((dataframes, relationships), df.target.values)

    assert isinstance(result, (float))


# cross_val_score cannot split entityset input
@pytest.mark.xfail
def test_sklearn_cross_val_score(df, es, pipeline):
    # Using with cross_val_score
    results = cross_val_score(
        pipeline, X=es, y=df.target.values, cv=2, scoring="accuracy"
    )

    assert isinstance(results[0], (float))
    assert isinstance(results[1], (float))


# GridSearchCV cannot split entityset input
@pytest.mark.xfail
def test_sklearn_gridsearchcv(df, es, pipeline):
    # Using with GridSearchCV
    params = {"et__max_depth": [5, 10]}
    grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=3)
    grid.fit(es, df.target.values)

    assert len(grid.predict(df["customer_id"].values)) == 15


def test_sklearn_cutoff_with_entityset(pipeline, es_customer_filtered):
    # Using cutoff_time to filter data
    ct = pd.DataFrame()
    ct["customer_id"] = [1, 2, 3]
    ct["time"] = pd.to_datetime(["2014-1-1 04:00", "2014-1-1 04:00", "2014-1-1 04:00"])
    ct["label"] = [True, True, False]

    results = pipeline.fit(X=(es_customer_filtered, ct), y=ct.label).predict(
        X=(es_customer_filtered, ct)
    )

    assert len(results) == 3


def test_sklearn_cutoff_with_dataframes_and_relationships(
    pipeline, es_customer_filtered
):
    # Using cutoff_time to filter data
    ct = pd.DataFrame()
    ct["customer_id"] = [1, 2, 3]
    ct["time"] = pd.to_datetime(["2014-1-1 04:00", "2014-1-1 04:00", "2014-1-1 04:00"])
    ct["label"] = [True, True, False]

    dataframes, relationships = get_dataframes_and_relationships(es_customer_filtered)
    results = pipeline.fit(X=((dataframes, relationships), ct), y=ct.label).predict(
        X=((dataframes, relationships), ct)
    )

    assert len(results) == 3


def test_cfm_uses_filtered_target_df_with_entityset(es):
    pipeline = Pipeline(
        steps=[("ft", DFSTransformer(target_dataframe_name="transactions"))]
    )

    train_ids = [1, 2, 3]
    test_ids = [10, 55, 853]

    train_es = filter_transactions(es, ids=train_ids)
    test_es = filter_transactions(es, ids=test_ids)

    fm_train = pipeline.fit_transform(X=train_es)
    assert all(fm_train["sessions.COUNT(transactions)"] == [1, 1, 1])
    assert set(fm_train.index.values) == set(train_ids)

    fm_test = pipeline.transform(test_es)

    assert all(fm_test["sessions.COUNT(transactions)"] == [1, 2, 2])
    assert set(fm_test.index.values) == set(test_ids)


def test_cfm_uses_filtered_target_df_with_dataframes_and_relationships(es):
    pipeline = Pipeline(
        steps=[("ft", DFSTransformer(target_dataframe_name="transactions"))]
    )

    train_ids = [3, 1, 2]
    test_ids = [853, 55, 10]

    train_es = filter_transactions(es, ids=train_ids)
    test_es = filter_transactions(es, ids=test_ids)
    train_dataframes, train_relationships = get_dataframes_and_relationships(train_es)
    test_dataframes, test_relationships = get_dataframes_and_relationships(test_es)

    fm_train = pipeline.fit_transform(X=(train_dataframes, train_relationships))
    assert all(fm_train["sessions.COUNT(transactions)"] == [1, 1, 1])
    assert set(fm_train.index.values) == set(train_ids)

    fm_test = pipeline.transform(X=(test_dataframes, test_relationships))
    assert all(fm_test["sessions.COUNT(transactions)"] == [2, 2, 1])
    assert set(fm_test.index.values) == set(test_ids)


def filter_transactions(es, ids):
    new_es = copy.deepcopy(es)
    customers_df = es["customers"]
    sessions_df = es["sessions"]
    products_df = es["products"]
    transactions_df = es["transactions"]
    transactions_df = transactions_df.loc[ids]
    sessions_df = sessions_df[
        sessions_df["session_id"].isin(transactions_df["session_id"].values)
    ]
    products_df = products_df[
        products_df["product_id"].isin(transactions_df["product_id"].values)
    ]
    customers_df = customers_df[
        customers_df["customer_id"].isin(sessions_df["customer_id"].values)
    ]
    new_es.replace_dataframe("customers", customers_df)
    new_es.replace_dataframe("sessions", sessions_df)
    new_es.replace_dataframe("transactions", transactions_df, already_sorted=True)
    new_es.replace_dataframe("products", products_df)

    return new_es
