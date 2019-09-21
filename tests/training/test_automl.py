# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd

from fklearn.training.automl import \
    automl_h2o_binary_classification_learner


def test_automl_h2o_binary_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    learner_binary = automl_h2o_binary_classification_learner(target="y")


    predict_fn_binary, pred_train_binary, log = learner_binary(df_train_binary)

    pred_test_binary = predict_fn_binary(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist()
    expected_col_test = df_test_binary.columns.tolist()

    assert Counter(expected_col_train) == Counter(pred_train_binary.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_binary.columns.tolist())
    assert pred_test_binary.prediction.max() < 1
    assert pred_test_binary.prediction.min() > 0
    assert (pred_test_binary.columns == pred_train_binary.columns).all()
