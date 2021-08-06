# featuretools-sklearn-transformer

![Tests](https://github.com/FeatureLabs/featuretools-sklearn-transformer/workflows/Tests/badge.svg)
[![Coverage Status](https://codecov.io/gh/alteryx/featuretools-sklearn-transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/alteryx/featuretools-sklearn-transformer)
[![PyPI version](https://badge.fury.io/py/featuretools-sklearn-transformer.svg?maxAge=2592000)](https://badge.fury.io/py/featuretools-sklearn-transformer)

[Featuretools](https://github.com/alteryx/featuretools)' DFS as a scikit-learn transformer

### Install
```shell
pip install featuretools_sklearn_transformer
```

### Use

To use the transformer in a pipeline, initialize an instance of the transformer by passing in
the parameters you would like to use for calculating features. To fit the model and generate features for
the training data, pass in an entityset or list of dataframes and relationships containing only the relevant
training data as the `X` input, along with the training targets as the `y` input. To generate a feature matrix from test data, pass in
an entityset containing only the relevant test data as the `X` input.

The input supplied for `X` can take several formats:
- To use a Featuretools EntitySet without cutoff times, simply pass in the EntitySet
- To use a Featuretools EntitySet with a cutoff times DataFrame, pass in a tuple of the form (EntitySet, cutoff_time_df)
- To use a list DataFrames and Relationships without cutoff times, pass a tuple of the form (dataframes, relationships)
- To use a list of DataFrames and Relationships with a cutoff times DataFrame, pass a tuple of the form ((dataframes, relationships), cutoff_time_df)

Note that because this transformer requires a Featuretools EntitySet or dataframes and relationships as input, it does not currently work
with certain methods such as `sklearn.model_selection.cross_val_score` or `sklearn.model_selection.GridSearchCV` which expect the `X` values
to be an iterable which can be split by the method.

The example below shows how to use the transformer with an EntitySet, both with and without a cutoff time DataFrame.

```python
import featuretools as ft
import pandas as pd

from featuretools.wrappers import DFSTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# Get example data
train_es = ft.demo.load_mock_customer(return_entityset=True, n_customers=3)
test_es = ft.demo.load_mock_customer(return_entityset=True, n_customers=2)
y = [True, False, True]

# Build pipeline
pipeline = Pipeline(steps=[
    ('ft', DFSTransformer(target_dataframe_name="customers",
                          max_features=2)),
    ('et', ExtraTreesClassifier(n_estimators=100))
])

# Fit and predict
pipeline.fit(X=train_es, y=y) # fit on customers in training entityset
pipeline.predict_proba(test_es) # predict probability of each class on test entityset
pipeline.predict(test_es) # predict on test entityset

# Same as above, but using cutoff times
train_ct = pd.DataFrame()
train_ct['customer_id'] = [1, 2, 3]
train_ct['time'] = pd.to_datetime(['2014-1-1 04:00',
                                   '2014-1-2 17:20',
                                   '2014-1-4 09:53'])

pipeline.fit(X=(train_es, train_ct), y=y)

test_ct = pd.DataFrame()
test_ct['customer_id'] = [1, 2]
test_ct['time'] = pd.to_datetime(['2014-1-4 13:48',
                                  '2014-1-5 15:32'])
pipeline.predict_proba((test_es, test_ct))
pipeline.predict((test_es, test_ct))
```

## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
