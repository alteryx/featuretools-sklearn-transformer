# featuretools-sklearn-transformer

[![CircleCI](https://circleci.com/gh/FeatureLabs/featuretools_sklearn_transformer/tree/master.svg?style=shield)](https://circleci.com/gh/FeatureLabs/featuretools_sklearn_transformer/tree/master)
[![codecov](https://codecov.io/gh/FeatureLabs/featuretools_sklearn_transformer/branch/master/graph/badge.svg)](https://codecov.io/gh/FeatureLabs/featuretools_sklearn_transformer)

Featuretools' DFS as a scikit-learn transformer

### Install
```shell
pip install featuretools_sklearn_transformer
```

### Use

```python
from featuretools.sklearn_transform import DFSTransformer

# Example Pipeline
pipeline = Pipeline(steps=[
    ('ft', DFSTransformer(entityset=es,
                          target_entity="customers",
                          max_features=20)),
    ("numeric", FunctionTransformer(select_numeric, validate=False)),
    ('imp', SimpleImputer()),
    ('et', ExtraTreesClassifier(n_estimators=10))
])

results = pipeline.fit(cutoff_time, y=cutoff_time.label).predict(cutoff_time)
```

## Feature Labs
<a href="https://www.featurelabs.com/">
    <img src="http://www.featurelabs.com/wp-content/uploads/2017/12/logo.png" alt="Featuretools" />
</a>

featuretools-sklearn-transformer is an open source project created by [Feature Labs](https://www.featurelabs.com/). To see the other open source projects we're working on visit Feature Labs [Open Source](https://www.featurelabs.com/open). If building impactful data science pipelines is important to you or your business, please [get in touch](https://www.featurelabs.com/contact/).
