# featuretools-sklearn-transformer

[![CircleCI](https://circleci.com/gh/FeatureLabs/featuretools-sklearn-transformer/tree/master.svg?style=shield)](https://circleci.com/gh/FeatureLabs/featuretools-sklearn-transformer/tree/master)
[![codecov](https://codecov.io/gh/FeatureLabs/featuretools-sklearn-transformer/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/featuretools-sklearn-transformer)

[Featuretools](https://github.com/FeatureLabs/featuretools)' DFS as a scikit-learn transformer

### Install
```shell
pip install featuretools_sklearn_transformer
```

### Use

To use the transformer in a pipeline, initialize an instance of the transformer by passing in
an entityset, or a list of entities and relationships. The input contain the complete set of data, including
both the training examples and the test examples. When calling `fit` or `transform`, simply pass in a
list of instance id values for the target entity that you would like to use in the pipeline. The transformer
will then create a feature matrix using only the specified instances.

```python
import featuretools as ft
import pandas as pd

from featuretools.wrappers import DFSTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# Get example data
n_customers = 3
es = ft.demo.load_mock_customer(return_entityset=True, n_customers=5)
y = [True, False, True]

# Build pipeline
pipeline = Pipeline(steps=[
    ('ft', DFSTransformer(entityset=es,
                            target_entity="customers",
                            max_features=2)),
    ('et', ExtraTreesClassifier(n_estimators=100))
])

# Fit and predict
pipeline.fit([1, 2, 3], y=y) # fit on first 3 customers
pipeline.predict_proba([4,5]) # predict probability of each class on last 2
pipeline.predict([4,5]) # predict on last 2

# Same as above, but using cutoff times
ct = pd.DataFrame()
ct['customer_id'] = [1, 2, 3, 4, 5]
ct['time'] = pd.to_datetime(['2014-1-1 04:00',
                                '2014-1-2 17:20',
                                '2014-1-4 09:53',
                                '2014-1-4 13:48',
                                '2014-1-5 15:32'])

pipeline.fit(ct.head(3), y=y)
pipeline.predict_proba(ct.tail(2))
pipeline.predict(ct.tail(2))
```

## Built at Alteryx Innovation Labs

<a href="https://www.alteryx.com/innovation-labs">
    <img src="https://evalml-web-images.s3.amazonaws.com/alteryx_innovation_labs.png" alt="Alteryx Innovation Labs" />
</a>
