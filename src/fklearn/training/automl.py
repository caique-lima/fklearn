from typing import List

import numpy as np
import pandas as pd
from toolz import curry, merge, assoc

from fklearn.types import LearnerReturnType, LogType
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time, expand_features_encoded


@curry
@log_learner_time(learner_name='automl_h2o_binary_classification_learner')
def automl_h2o_binary_classification_learner(df: pd.DataFrame,
                                             target: str,
                                             seed=42) -> LearnerReturnType:
    """
    Fits an logistic regression classifier to the dataset. Return the predict function
    for the model and the predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be discrete, since this is a classification model.

    params : dict
        The LogisticRegression parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    prediction_column : str
        The name of the column with the predictions from the model.
        If a multiclass problem, additional prediction_column_i columns will be added for i in range(0,n_classes).

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init()

    train = h2o.H2OFrame(df)

    x = train.columns
    y = target
    x.remove(y)

    train[y] = train[y].asfactor()

    aml = H2OAutoML(seed=seed, max_runtime_secs=60, nfolds=0, include_algos=["XGBoost"])
    aml.train(x=x, y=y, training_frame=train)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        data = h2o.H2OFrame(new_df)
        pred = aml.leader.predict(data).as_data_frame(use_pandas=True)
        new_df["prediction"] = pred["p1"]

        return new_df

    p.__doc__ = learner_pred_fn_docstring("automl_h2o_binary_classification_learner")

    log = {'automl_h2o_binary_classification_learner': {
        'target': target,
        'package': "h2o",
        'package_version': "3",
        'training_samples': len(df)},
        'object': aml}

    return p, p(df), log


automl_h2o_binary_classification_learner.__doc__ += learner_return_docstring("AutoML H2O Binary Classification Learner")
