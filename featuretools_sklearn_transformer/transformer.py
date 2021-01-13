from featuretools.computational_backends import calculate_feature_matrix
from featuretools.synthesis import dfs
from sklearn.base import TransformerMixin


class DFSTransformer(TransformerMixin):
    """Transformer using Scikit-Learn interface for Pipeline uses.
    """

    def __init__(self,
                 target_entity=None,
                 agg_primitives=None,
                 trans_primitives=None,
                 allowed_paths=None,
                 max_depth=2,
                 ignore_entities=None,
                 ignore_variables=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_primitives=None,
                 max_features=-1,
                 verbose=False):
        """Creates Transformer

        Args:

            entities (dict[str -> tuple(pd.DataFrame, str, str)]): Dictionary
                of entities. Entries take the format
                {entity id -> (dataframe, id column, (time_column))}.

            relationships (list[(str, str, str, str)]): List of relationships
                between entities. List items are a tuple with the format
                (parent entity id, parent variable, child entity id, child
                variable).

            entityset (EntitySet): An already initialized entityset. Required
                if entities and relationships are not defined.

            target_entity (str): Entity id of entity on which to make
                predictions.

            agg_primitives (list[str or AggregationPrimitive], optional): List
                of Aggregation Feature types to apply.

                    Default: ["sum", "std", "max", "skew", "min", "mean",
                              "count", "percent_true", "num_unique", "mode"]

            trans_primitives (list[str or TransformPrimitive], optional):
                List of Transform Feature functions to apply.

                    Default: ["day", "year", "month", "weekday", "haversine",
                              "num_words", "num_characters"]

            allowed_paths (list[list[str]]): Allowed entity paths on which to
                make features.

            max_depth (int) : Maximum allowed depth of features.

            ignore_entities (list[str], optional): List of entities to
                blacklist when creating features.

            ignore_variables (dict[str -> list[str]], optional): List of
                specific variables within each entity to blacklist when
                creating features.

            seed_features (list[:class:`.FeatureBase`]): List of manually
                defined features to use.

            drop_contains (list[str], optional): Drop features
                that contains these strings in name.

            drop_exact (list[str], optional): Drop features that
                exactly match these strings in name.

            where_primitives (list[str or PrimitiveBase], optional):
                List of Primitives names (or types) to apply with where
                clauses.

                    Default:

                        ["count"]

            max_features (int, optional) : Cap the number of generated features
                    to this number. If -1, no limit.

        Example:
            .. ipython:: python

                import featuretools as ft
                import pandas as pd

                from featuretools.wrappers import DFSTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import ExtraTreesClassifier

                # Get example data
                n_customers = 3
                es = ft.demo.load_mock_customer(return_entityset=True, n_customers=5)
                y = [True, False, True]

                # Build dataset
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

        """
        self.feature_defs = []
        self.target_entity = target_entity
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.allowed_paths = allowed_paths
        self.max_depth = max_depth
        self.ignore_entities = ignore_entities
        self.ignore_variables = ignore_variables
        self.seed_features = seed_features
        self.drop_contains = drop_contains
        self.drop_exact = drop_exact
        self.where_primitives = where_primitives
        self.max_features = max_features
        self.verbose = verbose

    def fit(self, X, y=None):
        """Wrapper for DFS

            Calculates a list of features given a dictionary of
            entities and a list of relationships. Alternatively,
            an EntitySet can be passed instead of the entities
            and relationships.

            See Also:
                :func:`synthesis.dfs`
        """
        es, entities, relationships, _ = parse_x_input(X)

        self.feature_defs = dfs(entityset=es,
                                entities=entities,
                                relationships=relationships,
                                target_entity=self.target_entity,
                                agg_primitives=self.agg_primitives,
                                trans_primitives=self.trans_primitives,
                                allowed_paths=self.allowed_paths,
                                max_depth=self.max_depth,
                                ignore_entities=self.ignore_entities,
                                ignore_variables=self.ignore_variables,
                                seed_features=self.seed_features,
                                drop_contains=self.drop_contains,
                                drop_exact=self.drop_exact,
                                where_primitives=self.where_primitives,
                                max_features=self.max_features,
                                features_only=True,
                                verbose=self.verbose)

        return self

    def transform(self, X):
        """Wrapper for calculate_feature_matrix

            Calculates a feature matrix for a the given set of instance ids and calculation
            times.

            Args:
                cutoff_time_ids (list | DataFrame): Instances filtered to
                    calculate features on.

            See Also:
                :func:`computational_backends.calculate_feature_matrix`
        """
        es, entities, relationships, cutoff_time = parse_x_input(X)

        X_transformed = calculate_feature_matrix(
            features=self.feature_defs,
            instance_ids=None,
            cutoff_time=cutoff_time,
            entityset=es,
            entities=entities,
            relationships=relationships,
            verbose=self.verbose)

        return X_transformed

    def get_params(self, deep=True):
        out = {
            'target_entity': self.target_entity,
            'agg_primitives': self.agg_primitives,
            'trans_primitives': self.trans_primitives,
            'allowed_paths': self.allowed_paths,
            'max_depth': self.max_depth,
            'ignore_entities': self.ignore_entities,
            'ignore_variables': self.ignore_variables,
            'seed_features': self.seed_features,
            'drop_contains': self.drop_contains,
            'drop_exact': self.drop_exact,
            'where_primitives': self.where_primitives,
            'max_features': self.max_features,
            'verbose': self.verbose,
        }
        return out


def parse_x_input(X):
    if isinstance(X, tuple):
        if isinstance(X[0], tuple):
            # Input of ((entities, relationships), cutoff_time)
            entities = X[0][0]
            relationships = X[0][1]
            es = None
            cutoff_time = X[1]
        elif isinstance(X[0], dict):
            # Input of (entities, relationships)
            entities = X[0]
            relationships = X[1]
            es = None
            cutoff_time = None
        else:
            # Input of (entityset, cutoff_time)
            es = X[0]
            entities = None
            relationships = None
            cutoff_time = X[1]
    else:
        # Input of entityset
        es = X
        entities = None
        relationships = None
        cutoff_time = None

    return es, entities, relationships, cutoff_time
