import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.6445642407906559
exported_pipeline = make_pipeline(
    FeatureAgglomeration(affinity="euclidean", linkage="ward"),
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=False, l1_ratio=0.5, learning_rate="constant", loss="perceptron", penalty="elasticnet", power_t=0.5)),
    StackingEstimator(estimator=MLPClassifier(alpha=0.001, learning_rate_init=1.0)),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.3, min_samples_leaf=7, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
