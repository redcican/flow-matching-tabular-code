"""Machine learning efficacy (MLE) via train-on-synthetic-test-on-real evaluation.

MLE = 1 - |Acc_real - Acc_synthetic| / Acc_real

Models trained on synthetic data and evaluated on real test sets using
logistic regression, random forest, and XGBoost classifiers.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def compute_mle(real_train: np.ndarray, real_test: np.ndarray,
                synthetic_train: np.ndarray,
                real_labels_train: np.ndarray, real_labels_test: np.ndarray,
                synthetic_labels: np.ndarray) -> dict:
    """Compute Machine Learning Efficacy score.

    Args:
        real_train: real training features
        real_test: real test features
        synthetic_train: synthetic training features
        real_labels_train: real training labels
        real_labels_test: real test labels
        synthetic_labels: synthetic training labels

    Returns:
        dict with per-model MLE scores and average
    """
    # Encode labels
    le = LabelEncoder()
    le.fit(np.concatenate([real_labels_train, real_labels_test, synthetic_labels]))
    y_real_train = le.transform(real_labels_train)
    y_real_test = le.transform(real_labels_test)
    y_syn_train = le.transform(synthetic_labels)

    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for name, clf in classifiers.items():
        # Train on real, test on real (baseline accuracy)
        clf_real = type(clf)(**clf.get_params())
        clf_real.fit(real_train, y_real_train)
        acc_real = accuracy_score(y_real_test, clf_real.predict(real_test))

        # Train on synthetic, test on real
        clf_syn = type(clf)(**clf.get_params())
        clf_syn.fit(synthetic_train, y_syn_train)
        acc_syn = accuracy_score(y_real_test, clf_syn.predict(real_test))

        # MLE = 1 - |acc_real - acc_syn| / acc_real
        mle = 1.0 - abs(acc_real - acc_syn) / max(acc_real, 1e-8)
        results[name] = {
            "acc_real": acc_real,
            "acc_synthetic": acc_syn,
            "mle": max(0.0, mle),
        }

    avg_mle = np.mean([r["mle"] for r in results.values()])
    results["average_mle"] = avg_mle

    return results
