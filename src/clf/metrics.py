# base calibration metric

# https://github.com/JonathanWenger/pycalib/blob/master/pycalib/scoring.py
# https://github.com/google-research/robustness_metrics/blob/master/robustness_metrics/metrics/uncertainty.py
#  https://github.com/kjdhfg/fd-shifts

from __future__ import annotations
import scipy
import sklearn.utils.validation

from dataclasses import dataclass
from functools import cached_property
from typing import Any
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn import metrics as skm
import evaluate as HF_evaluate

ArrayType = npt.NDArray[np.floating]

from collections import OrderedDict

## https://github.com/IML-DKFZ/fd-shifts/blob/main/fd_shifts/analysis/confid_scores.py#L20

# ----------------------------------- general metrics with consistent metric(y_true, p_hat) API -----------------------------------


def f1_micro(y_true, p_hat, y_hat=None):
    if y_hat is None:
        y_hat = np.argmax(p_hat, axis=-1)
    return skm.f1_score(y_true, y_hat, average="micro")


def f1_macro(y_true, p_hat, y_hat=None):
    if y_hat is None:
        y_hat = np.argmax(p_hat, axis=-1)
    return skm.f1_score(y_true, y_hat, average="macro")


# Pure numpy and TF implementations of proper losses (as metrics) -----------------------------------


def brier_loss(y_true, p_hat):
    r"""Brier score.
    If the true label is k, while the predicted vector of probabilities is
    [y_1, ..., y_n], then the Brier score is equal to
    \sum_{i != k} y_i^2 + (y_k - 1)^2.

    The smaller the Brier score, the better, hence the naming with "loss".
    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). The Brier loss is composed of refinement loss and
    calibration loss.

    """
    N = len(y_true)
    K = p_hat.shape[-1]

    if y_true.shape != p_hat.shape:
        zeros = scipy.sparse.lil_matrix((N, K))
        for i in range(N):
            zeros[i, y_true[i]] = 1

    if not np.isclose(np.sum(p_hat), len(p_hat)):
        p_hat = scipy.special.softmax(p_hat, axis=-1)

    return np.mean(np.sum(np.array(p_hat - zeros) ** 2, axis=1))


def nll(y_true, p_hat):
    r"""Multi-class negative log likelihood.
    If the true label is k, while the predicted vector of probabilities is
    [p_1, ..., p_K], then the negative log likelihood is -log(p_k).
    Does not require onehot encoding
    """
    labels = np.arange(p_hat.shape[-1])
    return skm.log_loss(y_true, p_hat, labels=labels)


def accuracy(y_true, p_hat):
    y_pred = np.argmax(p_hat, axis=-1)
    return sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def error(y_true, p_hat):
    return 1 - accuracy(y_true=y_true, p_hat=p_hat)


def odds_correctness(y_true, p_hat):
    """
    Computes the odds of making a correct prediction.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    p_hat : array-like
        Array of confidence estimates.

    Returns
    -------
    odds : float
    """
    return accuracy(y_true=y_true, p_hat=p_hat) / error(y_true=y_true, p_hat=p_hat)


def sharpness(y, p_hat, ddof=1):
    """
    Computes the empirical sharpness of a classifier.

    Computes the empirical sharpness of a classifier by computing the sample variance of a
    vector of confidence estimates.

    Parameters
    ----------
    y : array-like
        Ground truth labels. Dummy argument for consistent cross validation.
    p_hat : array-like
        Array of confidence estimates
    ddof : int, optional, default=1
        Degrees of freedom for the variance estimator.

    Returns
    -------
    float
        Sharpness
    """
    # Number of classes
    n_classes = np.shape(p_hat)[-1]

    # Find prediction confidence
    p_max = np.max(p_hat, axis=1)

    # Compute sharpness
    sharp = np.var(p_max, ddof=ddof) * 4 * n_classes**2 / (n_classes - 1) ** 2

    return sharp


def overconfidence(y, p_hat):
    """
    Computes the overconfidence of a classifier.

    Computes the empirical overconfidence of a classifier on a test sample by evaluating
    the average confidence on the false predictions.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_hat : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Overconfidence
    """
    # Find prediction and confidence
    y_pred = np.argmax(p_hat, axis=1)
    p_max = np.max(p_hat, axis=1)

    return np.average(p_max[y_pred != y])


def underconfidence(y, p_hat):
    """
    Computes the underconfidence of a classifier.

    Computes the empirical underconfidence of a classifier on a test sample by evaluating
    the average uncertainty on the correct predictions.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_hat : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Underconfidence
    """
    # Find prediction and confidence
    y_pred = np.argmax(p_hat, axis=1)
    p_max = np.max(p_hat, axis=1)

    return np.average(1 - p_max[y_pred == y])


def ratio_over_underconfidence(y, p_hat):
    """
    Computes the ratio of over- and underconfidence of a classifier.

    Computes the empirical ratio of over- and underconfidence of a classifier on a test sample.

    Parameters
    ----------
    y : array-like
        Ground truth labels
    p_hat : array-like
        Array of confidence estimates

    Returns
    -------
    float
        Ratio of over- and underconfidence
    """
    return overconfidence(y=y, p_hat=p_hat) / underconfidence(y=y, p_hat=p_hat)


def average_confidence(y, p_hat):
    """
    Computes the average confidence in the prediction

    Parameters
    ----------
    y : array-like
        Ground truth labels. Here a dummy variable for cross validation.
    p_hat : array-like
        Array of confidence estimates.

    Returns
    -------
    avg_conf:float
        Average confidence in prediction.
    """
    return np.mean(np.max(p_hat, axis=1))


def weighted_abs_conf_difference(y, p_hat):
    """
        Computes the weighted absolute difference between over and underconfidence.
    aurc, cache
        Parameters
        ----------
        y : array-like
            Ground truth labels. Here a dummy variable for cross validation.
        p_hat : array-like
            Array of confidence estimates.

        Returns
        -------
        weighted_abs_diff: float
            Accuracy weighted absolute difference between over and underconfidence.
    """
    y_pred = np.argmax(p_hat, axis=1)
    of = overconfidence(y, p_hat)
    uf = underconfidence(y, p_hat)

    return abs((1 - np.average(y == y_pred)) * of - np.average(y == y_pred) * uf)


def precision(y, p_hat, **kwargs):
    """
    Computes the precision.

    Parameters
    ----------
    y
    p_hat

    Returns
    -------

    """
    y_pred = np.argmax(p_hat, axis=1)
    return sklearn.metrics.precision_score(y_true=y, y_pred=y_pred, **kwargs)


def recall(y, p_hat, **kwargs):
    """
    Computes the recall.

    Parameters
    ----------
    y
    p_hat

    Returns
    -------

    """
    y_pred = np.argmax(p_hat, axis=1)
    return sklearn.metrics.recall_score(y_true=y, y_pred=y_pred, **kwargs)


AURC_DISPLAY_SCALE = 1  # 1000

"""
From: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/reports/custom/report52.pdf 

The risk-coverage (RC) curve [28, 16] is a measure of the trade-off between the
coverage (the proportion of test data encountered), and the risk (the error rate under this coverage). Since each
prediction comes with a confidence score, given a list of prediction correctness Z paired up with the confidence
scores C, we sort C in reverse order to obtain sorted C'
, and its corresponding correctness Z'
. Note that the correctness is computed based on Exact Match (EM) as described in [22]. The RC curve is then obtained by
computing the risk of the coverage from the beginning of Z'
(most confident) to the end (least confident). In particular, these metrics evaluate 
the relative order of the confidence score, which means that we want wrong
answers have lower confidence score than the correct ones, ignoring their absolute values. 

Source: https://github.com/kjdhfg/fd-shifts 

References:
-----------

[1] Jaeger, P.F., Lüth, C.T., Klein, L. and Bungert, T.J., 2022. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. arXiv preprint arXiv:2211.15259.

[2] Kamath, A., Jia, R. and Liang, P., 2020. Selective Question Answering under Domain Shift. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (pp. 5684-5696).

"""


@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values
        correct (array_like): Boolean array (best converted to int) where predictions were correct
    """

    confids: npt.NDArray[Any]
    correct: npt.NDArray[Any]

    @cached_property
    def roc_curve_stats(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        fpr, tpr, _ = skm.roc_curve(self.correct, self.confids)
        return fpr, tpr

    @property
    def residuals(self) -> npt.NDArray[Any]:
        return 1 - self.correct

    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        risks = []

        n_residuals = len(self.residuals)
        idx_sorted = np.argsort(self.confids)

        coverage = n_residuals
        error_sum = sum(self.residuals[idx_sorted])

        coverages.append(coverage / n_residuals)
        risks.append(error_sum / n_residuals)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.residuals[idx_sorted[i]]
            selective_risk = error_sum / (n_residuals - 1 - i)
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_residuals)
                risks.append(selective_risk)
                weights.append(tmp_weight / n_residuals)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            risks.append(risks[-1])
            weights.append(tmp_weight / n_residuals)
        return coverages, risks, weights


def AUROC_PR(pred_known, pred_unknown):
    neg = list(np.max(pred_known, axis=-1))
    pos = list(np.max(pred_unknown, axis=-1))
    auroc, aupr = compute_auc_aupr(neg, pos, pos_label=0)
    return auroc, aupr


def compute_auc_aupr(neg, pos, pos_label=1):  # zeros are known; ones are unknown
    ys = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))), axis=0)
    neg = np.array(neg)[np.logical_not(np.isnan(neg))]
    pos = np.array(pos)[np.logical_not(np.isnan(pos))]
    scores = np.concatenate((neg, pos), axis=0)
    auc = skm.roc_auc_score(ys, scores)  # AUROC ##1 as default
    aupr = skm.average_precision_score(ys, scores)  # AUPR
    if pos_label == 1:
        return auc, aupr
    else:
        return 1 - auc, 1 - aupr


def failauc(stats_cache: StatsCache) -> float:
    """AUROC_f metric function
    Args:
        stats_cache (StatsCache): StatsCache object
    Returns:
        metric value
    """
    fpr, tpr = stats_cache.roc_curve_stats
    return skm.auc(fpr, tpr)


def aurc(stats_cache: StatsCache):
    """auc metric function
    Args:
        stats_cache (StatsCache): StatsCache object
    Returns:
        metric value
    Important for assessment: LOWER is better!
    """
    _, risks, weights = stats_cache.rc_curve_stats
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]) * AURC_DISPLAY_SCALE


def aurc_logits(references, predictions, plot=False, get_cache=False, use_as_is=False):
    if not use_as_is:
        if not np.isclose(np.sum(references), len(references)):
            references = (np.argmax(predictions, -1) == references).astype(int)  # correctness

        if not np.isclose(np.sum(predictions), len(predictions)):
            predictions = scipy.special.softmax(predictions, axis=-1)

        if predictions.ndim == 2:
            predictions = np.max(predictions, -1)

    cache = StatsCache(confids=predictions, correct=references)

    if plot:
        coverages, risks, weights = cache.rc_curve_stats
        pd.options.plotting.backend = "plotly"
        df = pd.DataFrame(zip(coverages, risks, weights), columns=["% Coverage", "% Risk", "weights"])
        fig = df.plot(x="% Coverage", y="% Risk")
        fig.show()
    if get_cache:
        return {"aurc": aurc(cache), "cache": cache}
    return aurc(cache)




def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy
    Uses predictive entropy with T thresholds.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''
    def categorical_entropy(p):
        """Entropy of categorical distribution.
        Arguments:
            p: (B, d)

        Returns:
            (B,)
        """
        eps = 1e-12
        return -np.sum(p * np.log(p + eps), axis=-1)
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores_id, threshold)
        mask = np.array(scores_id <= p)
        thresholded_accuracies.append(np.mean(accuracies_test[mask]))
    values_id = np.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id

def multi_aurc_plot(caches, names, aurcs=None, verbose=False):
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame()
    for cache, name in zip(caches, names):
        coverages, risks, weights = cache.rc_curve_stats
        df[name] = pd.Series(risks, index=coverages)
    if verbose:
        print(df.head(), df.index, df.columns)
    fig = df.plot()
    title = ""
    if aurcs is not None:
        title = "AURC: " + " - ".join([str(round(aurc, 4)) for aurc in aurcs])
    fig.update_layout(title=title, xaxis_title="% Coverage", yaxis_title="% Risk")
    fig.show()


def AUROC_logits(references, predictions):
    if not np.isclose(np.sum(predictions), len(predictions)):
        predictions = scipy.special.softmax(predictions, axis=-1)

    cache = StatsCache(confids=predictions, correct=references)
    return {"AUROC": failauc(cache)}


def ece_logits(references, predictions):
    if not np.isclose(np.sum(predictions), len(predictions)):
        predictions = scipy.special.softmax(predictions, axis=-1)

    metric = HF_evaluate.load("jordyvl/ece")
    kwargs = dict(
        n_bins=min(len(predictions) - 1, 100),
        scheme="equal-mass",
        bin_range=[0, 1],
        proxy="upper-edge",
        p=1,
        detail=False,
    )

    ece_result = metric.compute(
        references=references,
        predictions=predictions,
        **kwargs,
    )
    return ece_result["ECE"]


def test_aurc():
    """
    Three cases from https://openreview.net/pdf?id=YnkGMIh0gvX

    separable_less_accurate_references ; acc 2/5, AUROC 1
    unseparable_lowcorrect_references ; acc 3/5, AUROC 0.75
    unseparable_highincorrect_references ; acc 3/5, AUROC 0.583
    """
    predictions = np.array([0.9, 0.1, 0.3, 1.0, 0.1])
    separable_less_accurate_references = np.array([1, 0, 0, 1, 0])
    result = aurc_logits(separable_less_accurate_references, predictions)
    print(f"separable_less_accurate gives an AURC of {result}")

    unseparable_lowcorrect_references = np.array([1, 1, 0, 1, 0])
    result = aurc_logits(unseparable_lowcorrect_references, predictions)
    print(f"unseparable_lowcorrect gives an AURC of {result}")  # BEST!

    unseparable_highincorrect_references = np.array([0, 1, 1, 1, 0])
    result = aurc_logits(unseparable_highincorrect_references, predictions)
    print(f"unseparable_highincorrect gives an AURC of {result}")


def test_ood():
    """
    Simple example following methodology in https://ieeexplore.ieee.org/document/9761166

    * Reversed labeling of IID vs OOD, just use pos_label=1
    """
    gt = [1, 0, 1, 0, 1, 1, 1, 1, 0]  # 1 being IID, 0 being IID
    predictions = [
        0.6648081,
        0.98290163,
        0.79909354,
        0.9961113,
        0.1472904,
        0.29210454,
        0.0049987,
        0.70650965,
        0.97676945,
    ]
    result = AUROC_logits(gt, predictions)
    print(f"worst AUROC_ood of {result}")
    """
    # p_iid = [0.66, 0.8, 0.14, 0.29, 0.004, 0.7]
    # p_ood = [0.98, 0.99, 0.97] 
    #indeed AUROC is 0 -> ood ranked higher than iid
    """
    gt = [0, 1, 0, 1, 1, 1, 1, 1, 0]  #
    result = AUROC_logits(gt, predictions)
    print(f"mixed AUROC_ood of {result}")

    gt = np.logical_not([1, 0, 1, 0, 1, 1, 1, 1, 0])  # 1 being IID
    result = AUROC_logits(gt, predictions)
    print(f"perfect AUROC_ood of {result}")


METRICS = [accuracy, brier_loss, nll, f1_micro, f1_macro, ece_logits, aurc_logits]


def apply_metrics(y_true, y_probs, metrics=METRICS):
    predictive_performance = OrderedDict()
    for metric in metrics:
        try:
            predictive_performance[f"{metric.__name__.replace('_logits', '')}"] = metric(y_true, y_probs)
        except Exception as e:
            print(e)
    # print(json.dumps(predictive_performance, indent=4))
    return predictive_performance


def evaluate_coverages(logits, labels, confidence, coverages=[100, 99, 98, 97, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10]):
    
    correctness = np.equal(logits.argmax(-1), labels)
    abstention_results = list(zip(list( confidence), list(correctness)))
    # sort the abstention results according to their reservations, from high to low
    abstention_results.sort(key = lambda x: x[0])
    # get the "correct or not" list for the sorted results
    sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
    size = len(sorted_correct)
    print('Abstention Logit: accuracy of coverage ') #1-risk
    for coverage in coverages:
        covered_correct = sorted_correct[:round(size/100*coverage)]
        print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
    print('')

    sr_results = list(zip(list(logits.max(-1)), list(correctness)))
    # sort the abstention results according to Softmax Response scores, from high to low
    sr_results.sort(key = lambda x: -x[0])
    # get the "correct or not" list for the sorted results
    sorted_correct = list(map(lambda x: int(x[1]), sr_results))
    size = len(sorted_correct)
    print('Softmax Response: accuracy of coverage ')
    for coverage in coverages:
        covered_correct = sorted_correct[:round(size/100*coverage)]
        print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
    print('')    

def compute_metrics(eval_preds):
    logits, labels = eval_preds  # output of forward
    if isinstance(logits, tuple):
        confidence = logits[1]
        logits = logits[0]
        if confidence.size == logits.shape[0]:
            evaluate_coverages(logits, labels, confidence)
    results = apply_metrics(labels, logits)
    return results


if __name__ == "__main__":
    pass
    # test_aurc()
    # test_ood()
