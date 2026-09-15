"""
Variants of FuzzyEspecialistEnsembleClassifier for the comparative analysis.

Base_Model.py is intentionally never edited here -- each class below is a
subclass overriding exactly one seam, so the original stays the fixed
reference point for comparison.

Section 1 (clustering algorithm): each class changes only the fuzzy
clustering algorithm itself, by overriding `_fit_fuzzy_clustering` -- the seam
both `get_best_k` (evaluating candidate k values) and `run_fuzzy_clustering`
(the final clustering used for training) call through. Everything else --
k-selection voting, membership-weighted bootstrap sampling, per-cluster tree
training, and distance-weighted fusion at inference -- is inherited unchanged.
All three still return a (centers, membership) pair in the base class's
contract shape:
    centers:    ndarray, shape (k, n_features)
    membership: ndarray, shape (k, n_samples)

Section 2 (k-selection speed): changes only how `k` is searched for, by
overriding `get_best_k` -- the clustering algorithm, sampling, training, and
fusion are all untouched.
"""

import numpy as np
import skfuzzy as fuzz
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

from .Base_Model import (
    FuzzyEspecialistEnsembleClassifier,
    FuzzyClusteringMetrics,
    DEFAULT_FUZZY_M,
    DEFAULT_FUZZY_ERROR,
    DEFAULT_FUZZY_MAX_ITER,
    DEFAULT_K_RANGE,
    DEFAULT_FALLBACK_K,
    DEFAULT_MEMBERSHIP_THRESHOLD,
    DEFAULT_NUM_SUBSAMPLES,
    DEFAULT_MIN_SAMPLES,
)


# ==========================================
# 1. Gaussian Mixture Model (soft clustering via posterior probabilities)
# ==========================================
class GMMEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with a Gaussian Mixture Model. A GMM is already
    "fuzzy" in the sense we need: `predict_proba` gives each point a soft
    responsibility (posterior probability) per component, which plays the
    same role fuzzy c-means' membership degree does everywhere downstream.

    Note: `fuzzy_m` (the fuzziness exponent) has no equivalent in a GMM and
    is ignored here. `fuzzy_error` is passed through as GaussianMixture's
    `tol` (convergence tolerance) -- a different quantity than fuzzy c-means'
    membership-shift tolerance, but serving the same "when do we stop
    iterating" role.
    """

    def _fit_fuzzy_clustering(self, X_T, k, m=None, error=None, max_iter=None):
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        X = X_T.T  # GaussianMixture expects (n_samples, n_features)
        gmm = GaussianMixture(
            n_components=k,
            tol=error,
            max_iter=max_iter,
            random_state=self.random_state,
        )
        gmm.fit(X)

        membership = gmm.predict_proba(X).T  # (n_samples, k) -> (k, n_samples)
        centers = gmm.means_                  # (k, n_features)
        return centers, membership


# ==========================================
# 2. Fuzzy c-medoids (cluster prototypes are real data points, not means)
# ==========================================
def _fuzzy_c_medoids(X, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    Fuzzy c-medoids (FCMdd): same iterative membership-then-prototype-update
    structure as fuzzy c-means, except each cluster's prototype is required
    to be an actual data point (a medoid) rather than a computed mean --
    useful when the mean of a cluster isn't a meaningful point, or when you
    want prototypes that are guaranteed real, interpretable examples.

    Returns (centers, membership, medoid_indices):
        centers:    ndarray, shape (c, n_features) -- the chosen medoids
        membership: ndarray, shape (c, n_samples)
        medoid_indices: ndarray, shape (c,) -- row indices into X
    """
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)

    dist_matrix = cdist(X, X, metric='euclidean')
    medoid_idx = rng.choice(n_samples, size=c, replace=False)

    u = np.full((c, n_samples), 1.0 / c)
    power = 2.0 / (m - 1)

    for _ in range(max_iter):
        d = np.fmax(dist_matrix[medoid_idx], 1e-10)  # (c, n_samples)
        inv_d = d ** (-power)
        u_new = inv_d / inv_d.sum(axis=0, keepdims=True)

        # Pick, for each cluster, the candidate point minimizing the
        # membership-weighted sum of distances to every other point.
        weights = u_new ** m
        costs = weights @ dist_matrix  # (c, n_samples); dist_matrix is symmetric
        new_medoid_idx = np.argmin(costs, axis=1)

        shift = np.max(np.abs(u_new - u))
        u = u_new
        medoids_changed = not np.array_equal(new_medoid_idx, medoid_idx)
        medoid_idx = new_medoid_idx

        if shift < error and not medoids_changed:
            break

    centers = X[medoid_idx]
    return centers, u, medoid_idx


class FuzzyKMedoidsEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with fuzzy c-medoids. `fuzzy_m` and `fuzzy_error`
    keep their usual meaning (fuzziness exponent, membership-shift
    convergence tolerance); medoid search uses `random_state` only for its
    initial medoid pick, since the update rule itself is deterministic given
    that start.
    """

    def _fit_fuzzy_clustering(self, X_T, k, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        X = X_T.T  # (n_samples, n_features)
        centers, membership, _ = _fuzzy_c_medoids(
            X, c=k, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, membership


# ==========================================
# 3. Possibilistic C-Means (typicalities, not memberships -- rows need not
#    sum to 1, so a point can belong "a lot" to several clusters or "a
#    little" to all of them; better suited to noisy data / outlier handling)
# ==========================================
def _possibilistic_cmeans(X_T, c, m=2, error=0.005, max_iter=1000, random_state=None):
    """
    Possibilistic C-Means (Krishnapuram & Keller, 1993). Bootstraps from a
    standard fuzzy c-means run to get initial centers and to estimate each
    cluster's typicality scale (eta) -- the common way to avoid every
    cluster collapsing onto the same point, which is PCM's well-known
    failure mode without a good initialization.

    Returns (centers, typicality):
        centers:    ndarray, shape (c, n_features)
        typicality: ndarray, shape (c, n_samples) -- NOT column-normalized;
                    unlike fuzzy c-means/c-medoids/GMM, a point's values
                    across clusters need not sum to 1.
    """
    cntr0, u0, _, _, _, _, _ = fuzz.cmeans(
        data=X_T, c=c, m=m, error=error, maxiter=max_iter, init=None
    )
    X = X_T.T  # (n_samples, n_features)

    d0_sq = cdist(X, cntr0, metric='sqeuclidean').T  # (c, n_samples)
    weights0 = u0 ** m
    eta = (weights0 * d0_sq).sum(axis=1) / weights0.sum(axis=1)
    eta = np.fmax(eta, 1e-10)

    centers = cntr0.copy()
    power = 1.0 / (m - 1)

    for _ in range(max_iter):
        d_sq = cdist(X, centers, metric='sqeuclidean').T  # (c, n_samples)
        t = 1.0 / (1.0 + (d_sq / eta[:, None]) ** power)

        weights = t ** m
        new_centers = (weights @ X) / weights.sum(axis=1, keepdims=True)

        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < error:
            break

    return centers, t


class PossibilisticCMeansEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces fuzzy c-means with possibilistic c-means (PCM).

    Worth knowing before using this one: PCM's typicalities deliberately
    don't sum to 1 across clusters for a point, which is exactly what makes
    it more robust to noise/outliers -- but it also means the k-selection
    metrics that assume a normalized partition (FPC, MPC, FPE in
    FuzzyClusteringMetrics) are less meaningful here than the hard-label-
    based ones (Silhouette, Calinski-Harabasz, Davies-Bouldin, the elbow
    method), which only depend on each point's argmax cluster and stay valid
    regardless of normalization. `membership_threshold` (default 0.3) was
    tuned with fuzzy c-means' normalized memberships in mind and may need
    re-tuning for PCM's typicality scale.
    """

    def _fit_fuzzy_clustering(self, X_T, k, m=None, error=None, max_iter=None):
        m = self.fuzzy_m if m is None else m
        error = self.fuzzy_error if error is None else error
        max_iter = self.fuzzy_max_iter if max_iter is None else max_iter

        centers, typicality = _possibilistic_cmeans(
            X_T, c=k, m=m, error=error, max_iter=max_iter, random_state=self.random_state
        )
        return centers, typicality


# ==========================================
# 4. Faster k-selection: fewer metrics
# ==========================================
class FastKSelectionEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Same clustering algorithm (fuzzy c-means), sampling, training, and fusion
    as the base class -- the only thing this changes is how expensive
    picking `k` is: only 3 of the original 10 cluster-validity metrics are
    computed per candidate k, instead of all 10. Concretely:

        - FPC (Fuzzy Partition Coefficient): kept because it's the fuzzy-
          native index in the original set that's also cheap (O(n*k), no
          pairwise distance matrix) -- it measures how "crisp" vs "fuzzy"
          the partition is, which none of the hard-label metrics can see.
        - Xie-Beni: kept because it reuses the same cached (n_samples, k)
          distances FPC's cluster computes, so it's essentially free once
          FPC is available, and it's one of the few validity indices that
          explicitly accounts for the fuzziness exponent `m`.
        - Calinski-Harabasz: kept as the cheapest *hard-label* index (a
          ratio of between/within-cluster dispersion, no pairwise distance
          matrix needed) -- a sanity check against FPC/Xie-Beni agreeing
          with each other for reasons specific to being fuzzy-only
          measures, with no notion of separation in the original feature
          space.

    Silhouette, Davies-Bouldin, Dunn, MPC, FPE, Fukuyama-Sugeno, and the
    elbow method are dropped. Silhouette specifically is why this matters:
    sklearn computes full pairwise distances for it, making it O(n^2) and,
    on anything but small datasets, almost certainly the single most
    expensive part of the original 10-metric committee.

    `_find_best_k_heuristic` is inherited unchanged -- it already skips any
    metric absent from its input columns, so restricting which metrics get
    computed automatically restricts the vote to just these three, without
    needing to touch the voting logic itself.

    Evaluating the candidate k values concurrently (a thread pool) was tried
    and measured, not just assumed: on this project's test hardware it made
    things slower before fixing an oversubscription issue (numpy's BLAS
    backend already parallelizes a single fuzzy c-means call internally, so
    stacking a thread pool on top without limiting BLAS to one thread per
    worker means the two layers fight for the same cores), and even after
    fixing that, it came out even with or slightly slower than plain serial
    execution once the metric reduction below was already in place -- the
    metric reduction was the actual win, not the threading. So this stays
    serial on purpose. If you're running this on hardware with many more
    cores than were used to test it, that trade-off may be worth
    re-measuring rather than assumed.

    One thing this does NOT change: `fuzz.cmeans` draws its own random
    initialization internally when not seeded, and the original `get_best_k`
    never seeded it either -- so candidate-k evaluations were already
    non-deterministic run to run before this change, independent of it.
    """

    def get_best_k(self, X_train_processed):
        X_train_values = np.asarray(X_train_processed)
        X_train_T = X_train_values.T

        results = []
        for k in self.k_range:
            try:
                cntr, u = self._fit_fuzzy_clustering(X_train_T, k)
                metrics_obj = FuzzyClusteringMetrics(X_train_values, u, cntr, self.fuzzy_m)
                # Only these three getters are called -- see the class
                # docstring for why these three. calculate_all() is
                # deliberately NOT used here: it would compute (and pay
                # for) all 10 metrics, including the O(n^2) Silhouette,
                # only to have most of them thrown away.
                results.append({
                    'k': k,
                    'FPC': metrics_obj.get_fpc(),
                    'Xie_Beni': metrics_obj.get_xie_beni(),
                    'Calinski_Harabasz': metrics_obj.get_calinski_harabasz(),
                })
            except Exception as e:
                print(f"An error occurred for k={k}: {e}")

        df_results = pd.DataFrame(results)
        self.k_optimization_results = df_results  # Store for plotting
        return self._find_best_k_heuristic(df_results)


# ==========================================
# 5. Base classifier: SVM instead of Decision Tree
# ==========================================
class SVMEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces the per-cluster DecisionTreeClassifier with an SVM (SVC).
    Everything else -- clustering, k-selection, sampling, fusion -- is
    inherited unchanged; only `_train_classifiers` is overridden.

    `probability=True` is required: without it, SVC has no `predict_proba`
    at all, and the base class's `predict_proba` (unchanged, inherited)
    calls `model.predict_proba(X_test)` on every per-cluster model. This
    makes fitting noticeably slower than the Decision Tree variant, since
    SVC then runs its own internal probability calibration on top of the
    normal fit -- worth knowing given `_train_classifiers` runs once per
    bootstrap subsample (`num_subsamples` times per `fit()` call, default
    100), not just once per cluster.

    `svm_C`, `svm_kernel`, and `svm_gamma` are exposed as constructor
    parameters (mirroring how the base class treats its own hyperparameters)
    so they're tunable via `get_params()`/`GridSearchCV` rather than fixed;
    add more of SVC's own kwargs the same way if you need them.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        svm_C: float = 1.0,
        svm_kernel: str = 'rbf',
        svm_gamma: str = 'scale',
    ):
        super().__init__(
            debug_folder=debug_folder,
            detailed_logs=detailed_logs,
            random_state=random_state,
            fuzzy_m=fuzzy_m,
            fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter,
            k_range=k_range,
            fallback_k=fallback_k,
            membership_threshold=membership_threshold,
            num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma

    def _train_classifiers(self, cluster_datasets):
        for cluster_name, data in cluster_datasets.items():
            cluster_idx = cluster_name.split('_')[1]
            base_cluster_key = f'cluster_{cluster_idx}'

            if base_cluster_key not in self.models:
                self.models[base_cluster_key] = []

            features = [c for c in data.columns if 'grau_pertencimento' not in str(c) and c != 'target']
            X = data[features]
            y = data['target']

            model = SVC(
                C=self.svm_C,
                kernel=self.svm_kernel,
                gamma=self.svm_gamma,
                probability=True,
                random_state=self.random_state,
            )
            model.fit(X, y)

            self.models[base_cluster_key].append(model)


# ==========================================
# 6. Base classifier: Perceptron instead of Decision Tree
# ==========================================
class _HardVoteProbaWrapper:
    """
    Wraps a classifier with no `predict_proba` (Perceptron has none at all --
    it's not a probabilistic model) so it still satisfies the
    (predict_proba, classes_) contract the base class's inherited
    `predict_proba` relies on, by turning a hard prediction into a one-hot
    vector.

    Chosen deliberately over probability calibration (e.g.
    `CalibratedClassifierCV`): calibration needs its own internal
    cross-validation, which is fragile exactly where this ensemble would
    need it most -- the per-cluster bootstrap subsamples `fit()` trains on
    can be small (`min_samples` defaults to 10), and a stratified internal
    CV split can fail outright if some class has fewer rows than folds.
    One-hot hard voting has no such failure mode, and it's also the
    standard way ensembles of unstable/weak linear classifiers like
    Perceptron are combined in the literature -- not a compromise made only
    for robustness.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        preds = self.base_estimator.predict(X)
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        probs = np.zeros((len(preds), len(self.classes_)))
        for row, pred in enumerate(preds):
            probs[row, class_to_idx[pred]] = 1.0
        return probs


class PerceptronEspecialistEnsembleClassifier(FuzzyEspecialistEnsembleClassifier):
    """
    Replaces the per-cluster DecisionTreeClassifier with a Perceptron,
    wrapped in `_HardVoteProbaWrapper` so it still plugs into the
    inherited, unchanged `predict_proba`/fusion logic. Only
    `_train_classifiers` is overridden -- clustering, k-selection,
    sampling, and fusion are all untouched.

    `perceptron_max_iter` and `perceptron_alpha` (L2 regularization
    strength) are exposed as constructor parameters; add more of
    Perceptron's own kwargs (e.g. `penalty`) the same way if needed.
    """

    def __init__(
        self,
        debug_folder: str = 'pre_oversampling',
        detailed_logs: bool = False,
        random_state: int = 42,
        fuzzy_m: float = DEFAULT_FUZZY_M,
        fuzzy_error: float = DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = DEFAULT_FUZZY_MAX_ITER,
        k_range: range = DEFAULT_K_RANGE,
        fallback_k: int = DEFAULT_FALLBACK_K,
        membership_threshold: float = DEFAULT_MEMBERSHIP_THRESHOLD,
        num_subsamples: int = DEFAULT_NUM_SUBSAMPLES,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        perceptron_max_iter: int = 1000,
        perceptron_alpha: float = 0.0001,
    ):
        super().__init__(
            debug_folder=debug_folder,
            detailed_logs=detailed_logs,
            random_state=random_state,
            fuzzy_m=fuzzy_m,
            fuzzy_error=fuzzy_error,
            fuzzy_max_iter=fuzzy_max_iter,
            k_range=k_range,
            fallback_k=fallback_k,
            membership_threshold=membership_threshold,
            num_subsamples=num_subsamples,
            min_samples=min_samples,
        )
        self.perceptron_max_iter = perceptron_max_iter
        self.perceptron_alpha = perceptron_alpha

    def _train_classifiers(self, cluster_datasets):
        for cluster_name, data in cluster_datasets.items():
            cluster_idx = cluster_name.split('_')[1]
            base_cluster_key = f'cluster_{cluster_idx}'

            if base_cluster_key not in self.models:
                self.models[base_cluster_key] = []

            features = [c for c in data.columns if 'grau_pertencimento' not in str(c) and c != 'target']
            X = data[features]
            y = data['target']

            base_model = Perceptron(
                max_iter=self.perceptron_max_iter,
                alpha=self.perceptron_alpha,
                random_state=self.random_state,
            )
            model = _HardVoteProbaWrapper(base_model)
            model.fit(X, y)

            self.models[base_cluster_key].append(model)
