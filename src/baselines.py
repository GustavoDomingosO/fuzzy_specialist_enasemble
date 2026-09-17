import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

# Mirrors Base_Model.py's DEFAULT_FUZZY_* constants (same numeric values),
# redefined locally rather than imported so baselines.py stays fully
# standalone and never depends on Base_Model.py.
_DEFAULT_FUZZY_M = 2
_DEFAULT_FUZZY_ERROR = 0.005
_DEFAULT_FUZZY_MAX_ITER = 1000


class EvaluateMixin:
    """
    Adds an `evaluate(X, y) -> (accuracy, auc)` method that mirrors
    `FuzzyEspecialistEnsembleClassifier.evaluate` exactly (same binary vs.
    multiclass AUC handling, same ValueError -> np.nan fallback with a
    printed warning), so every baseline reports metrics comparably to the
    fuzzy ensemble without needing to import Base_Model.py.

    Mix this in *before* the sklearn classifier base class so its MRO wins
    only for `evaluate` (the sklearn class still supplies `fit`, `predict`,
    `predict_proba`, `get_params`, etc.), e.g.:

        class RandomForestBaseline(EvaluateMixin, RandomForestClassifier):
            pass
    """

    def evaluate(self, X_test, y_test):
        # Ensure y_test is a 1D array
        if hasattr(y_test, "values"):
            y_true = y_test.values.ravel()
        else:
            y_true = np.asarray(y_test).ravel()

        # Get hard labels for Accuracy
        preds = self.predict(X_test)
        acc = accuracy_score(y_true, preds)

        # Get probabilities for AUC
        probs = self.predict_proba(X_test)

        try:
            # Handle Binary vs Multiclass properly
            if probs.shape[1] == 2:
                # Binary classification: pass only the probabilities of the positive class
                auc = roc_auc_score(y_true, probs[:, 1])
            else:
                # Multiclass classification: explicitly provide the labels
                # This prevents errors if a class is missing from this specific test fold
                labels = self.classes_ if hasattr(self, 'classes_') else np.unique(y_true)
                auc = roc_auc_score(y_true, probs, multi_class='ovr', labels=labels)

        except ValueError as e:
            # Print the error so you aren't flying blind if it fails
            print(f" AUC Warning in this fold: {e}")

            # If it's truly a 1-class fold, AUC remains mathematically impossible.
            auc = np.nan

        return acc, auc


# ==========================================
# Baselines
# ------------------------------------------
# Unlike model_variants.py, these are NOT subclasses of
# FuzzyEspecialistEnsembleClassifier — they're independent comparison
# models with their own native sklearn hyperparameters, only borrowing the
# `evaluate()` contract via EvaluateMixin so results line up with the
# ensemble's in the same experiment scripts.
# ==========================================

class RandomForestBaseline(EvaluateMixin, RandomForestClassifier):
    """
    Plain Random Forest baseline for comparison against the fuzzy ensemble.

    No custom __init__: this inherits RandomForestClassifier's constructor
    and every native hyperparameter (n_estimators, max_depth, criterion,
    ...) directly, so get_params()/set_params()/clone() work out of the box
    and the full sklearn RandomForest surface stays available/tunable.
    """
    pass


class ForestOfLocalTreesBaseline(EvaluateMixin, BaseEstimator, ClassifierMixin):
    """
    Reimplementation of "Forest of Local Trees" (FLT), Armano & Tamponi,
    Pattern Recognition, 2018 (doi:10.1016/j.patcog.2017.11.017).

    Unlike Bagging/Random Forest, there is no bootstrap resampling: every
    tree in the ensemble is trained on the FULL training set, but each
    sample is reweighted per tree according to a Gaussian-kernel distance
    from a tree-specific "centroid" (a training instance chosen via a
    stochastic, diversity-promoting picking rule). Diversity comes purely
    from this differential weighting, not from instance subsampling.

    Design decisions made (agreed with the user; see the discussion this
    class came out of, not just this docstring, for the full reasoning):

      - Centroid picking (paper's Eq. 1-2): a training instance is drawn
        according to a probability distribution over the dataset,
        initially uniform. After each pick, the picking probability of
        every remaining sample is multiplied by log(1 + distance to the
        centroid just picked) and renormalized -- a SEQUENTIAL, single
        -step multiplicative update relative only to the newest centroid
        (not a recomputation against the nearest of ALL centroids picked
        so far, which would instead be a k-means++-style rule).
      - Per-sample weights (Eq. 4) and the picking distances above are
        both computed in a common [0, 1]-normalized feature space (a
        MinMaxScaler fit once on the training data, V = identity in that
        space). This operationalizes the paper's "scaling matrix V used
        to normalize features in [0, 1]", which is not spelled out
        further -- this is our own concrete reading of it, applied
        consistently to both uses.
      - Each RDT is trained on the WHOLE weighted dataset (sample_weight),
        with a random subset of features considered at each split
        (`max_features`, matching the paper's n_c) and a cap on the total
        number of leaves in the tree (`max_leaf_nodes`, matching the
        paper's r_leaf; None reproduces r_leaf = infinity). sklearn's
        weighted Gini computation already matches the paper's Eq. 8
        (weighted class frequency) exactly. One acknowledged
        approximation: sklearn's `max_leaf_nodes` grows the tree in
        BEST-FIRST order (largest impurity decrease first), while the
        paper specifies literal BREADTH-FIRST (level-by-level) growth --
        both cap total leaf count, but the growth order differs.
      - The "percent of training samples" (zeta) ablation knob from the
        paper's experiments section is intentionally NOT implemented: it
        is absent from the paper's own Section 4 algorithm description,
        and their own recommended configuration uses zeta = 100% (i.e.,
        off) anyway. Skipped per explicit instruction to keep this
        faithful to Section 4 rather than the full parameter sweep.
      - `predict_proba` (the per-tree weighted vote turned into a
        normalized pseudo-probability) is OUR OWN ADDITION, not part of
        the paper -- the paper only defines a hard-label weighted
        -majority vote (Eq. 9) with no continuous score. Because of this,
        `evaluate()` intentionally WITHHOLDS AUC for this class (returns
        NaN with a printed note) until the probability construction is
        validated with the user's advisor. Do not use this model's AUC
        in experiments until that is resolved.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_features: float = 0.3,
        max_leaf_nodes: int = None,
        random_state: int = 42,
    ):
        # sklearn convention: __init__ only stores constructor arguments.
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y):
        X_values = np.asarray(X)
        y_values = np.asarray(y)
        n_samples = X_values.shape[0]

        self.classes_ = np.sort(pd.Series(y_values).unique())

        # Common [0, 1]-normalized feature space, shared by centroid
        # picking (Eq. 1-2) and sample weighting (Eq. 4) -- see class
        # docstring for why this reading was chosen.
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X_values)

        rng = np.random.default_rng(self.random_state)

        picking_probs = np.full(n_samples, 1.0 / n_samples)

        self.centroids_ = []
        self.trees_ = []

        for t in range(self.n_estimators):
            # --- Pick a new centroid (Eq. 1-2 setup) ---
            idx = rng.choice(n_samples, p=picking_probs)
            centroid = X_scaled[idx]
            self.centroids_.append(centroid)

            # --- Update picking probabilities (Eq. 1-2) ---
            dist_to_new_centroid = np.linalg.norm(X_scaled - centroid, axis=1)
            updated = picking_probs * np.log1p(dist_to_new_centroid)
            total = updated.sum()
            if total > 0:
                picking_probs = updated / total
            else:
                # All remaining probability mass collapsed to zero (e.g.
                # every remaining sample coincides exactly with a
                # previously picked centroid). Not covered by the paper;
                # fall back to uniform rather than crash.
                picking_probs = np.full(n_samples, 1.0 / n_samples)

            # --- Per-sample weights for this tree (Eq. 4, V = identity
            # in the scaled space) ---
            sq_dist = np.sum((X_scaled - centroid) ** 2, axis=1)
            sample_weight = np.exp(-0.5 * sq_dist)

            # --- Train the RDT on the FULL dataset with these weights ---
            tree_seed = self.random_state + t if self.random_state is not None else None
            tree = DecisionTreeClassifier(
                criterion="gini",
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=tree_seed,
            )
            tree.fit(X_values, y_values, sample_weight=sample_weight)
            self.trees_.append(tree)

        return self

    def _tree_weights(self, X_scaled):
        """
        Per-test-sample, per-tree competence weights (Eq. 4, evaluated
        against each tree's centroid instead of against training samples).
        """
        weights = np.zeros((X_scaled.shape[0], len(self.trees_)))
        for t, centroid in enumerate(self.centroids_):
            sq_dist = np.sum((X_scaled - centroid) ** 2, axis=1)
            weights[:, t] = np.exp(-0.5 * sq_dist)
        return weights

    def predict_proba(self, X):
        """
        NOTE: not part of Armano & Tamponi (2018) -- see class docstring.
        The paper only defines a hard-label weighted-majority vote
        (Eq. 9). This normalizes those same per-tree weights into a
        pseudo-probability so the class satisfies sklearn's classifier
        API, but it should not be treated as a validated probability
        estimate yet (hence `evaluate()` withholding AUC below).
        """
        check_is_fitted(self, attributes=["classes_"])

        X_values = np.asarray(X)
        X_scaled = self.scaler_.transform(X_values)
        n_samples = X_values.shape[0]
        n_classes = len(self.classes_)

        weights = self._tree_weights(X_scaled)  # (n_samples, n_trees)

        vote_sums = np.zeros((n_samples, n_classes))
        for t, tree in enumerate(self.trees_):
            tree_preds = tree.predict(X_values)
            for class_idx, cls in enumerate(self.classes_):
                vote_sums[:, class_idx] += weights[:, t] * (tree_preds == cls)

        totals = vote_sums.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0  # guard: only reachable in pathological cases
        return vote_sums / totals

    def predict(self, X):
        """
        Weighted-majority vote (Eq. 9), implemented via predict_proba's
        argmax -- equivalent, since normalizing does not change the argmax.
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def evaluate(self, X_test, y_test):
        """
        Returns (accuracy, auc). Accuracy is computed normally. AUC is
        intentionally WITHHELD (returned as NaN) for this baseline: the
        paper defines no continuous score, and predict_proba here is our
        own addition (see class docstring). Per explicit instruction, no
        AUC should be taken from this model in experiments until this is
        confirmed with the advisor.
        """
        if hasattr(y_test, "values"):
            y_true = y_test.values.ravel()
        else:
            y_true = np.asarray(y_test).ravel()

        preds = self.predict(X_test)
        acc = accuracy_score(y_true, preds)

        print(
            "AUC withheld for ForestOfLocalTreesBaseline: predict_proba is "
            "our own addition, not part of Armano & Tamponi (2018) -- see "
            "class docstring. Do not use it in experiments until confirmed."
        )
        return acc, np.nan


class FuzzyBaggingBaseline(EvaluateMixin, BaseEstimator, ClassifierMixin):
    """
    Reimplementation of "FuzzyBagging" (Nanni & Lumini, Pattern Recognition,
    2006, doi:10.1016/j.patcog.2005.10.002).

    Training set generation: Fuzzy C-Means is run ONCE on the whole
    training set, producing K (overlapping) clusters. For cluster i, a
    dedicated training subset is built by taking, PER CLASS separately,
    the 63.2% of that class's patterns with the highest membership to
    cluster i (63.2% = 1 - 1/e, chosen by the original authors purely to
    match Bagging's average per-sample inclusion rate -- it is not
    derived from the clustering itself). Because selection is membership
    -based rather than a hard partition, the same pattern can end up in
    more than one cluster's subset, hence "overlapping" clusters. One
    classifier is trained per cluster subset -- K classifiers total.

    Prediction: every test pattern is scored by ALL K classifiers (no
    test-time membership/routing at all, unlike the fuzzy ensemble this
    baseline is compared against), and their class-probability outputs
    are combined with the classic "max rule": for each class, take the
    largest probability any classifier assigned it, then predict the
    class with the largest resulting score. Note that, as is standard for
    the max rule, the returned scores need not sum to 1 across classes
    (each classifier's own row does, but taking an element-wise max
    across several such rows generally does not preserve that) -- this
    does not affect argmax-based prediction or rank-based AUC scoring.

    Two deliberate deviations from the paper (agreed with the user):

      - `base_estimator` defaults to a `DecisionTreeClassifier`. This is
        OUR OWN DECISION, made for consistency with the rest of this
        project's baseline suite -- the paper only ever tested a linear
        discriminant classifier (LDC) and the mechanism itself is not
        tied to any particular base classifier. Any sklearn-compatible
        classifier exposing `predict_proba` can be passed instead.
      - `min_samples_per_class` is a floor (default 1) on how many of a
        class's patterns are kept in each cluster's subset, added because
        the paper never addresses what happens when 63.2% of a class's
        count rounds to zero: without a floor, a rare class could vanish
        entirely from a given cluster's training subset, leaving that
        cluster's classifier unable to score it at test time. This floor
        is NOT part of the original paper.

    Left as literal, paper-specified free parameters (not gaps): the
    number of clusters K (`n_clusters`, swept as FB-3..FB-7 in the
    paper), and the 63.2% selection fraction itself (`selection_percent`,
    exposed here in case you want to deviate from it deliberately).
    """

    def __init__(
        self,
        n_clusters: int = 3,
        selection_percent: float = 0.632,
        min_samples_per_class: int = 1,
        base_estimator=None,
        fuzzy_m: float = _DEFAULT_FUZZY_M,
        fuzzy_error: float = _DEFAULT_FUZZY_ERROR,
        fuzzy_max_iter: int = _DEFAULT_FUZZY_MAX_ITER,
        random_state: int = 42,
    ):
        # sklearn convention: __init__ only stores constructor arguments.
        self.n_clusters = n_clusters
        self.selection_percent = selection_percent
        self.min_samples_per_class = min_samples_per_class
        self.base_estimator = base_estimator
        self.fuzzy_m = fuzzy_m
        self.fuzzy_error = fuzzy_error
        self.fuzzy_max_iter = fuzzy_max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X_values = np.asarray(X)
        y_values = np.asarray(y)

        self.classes_ = np.sort(pd.Series(y_values).unique())

        # --- Fuzzy C-Means on the WHOLE training set, once ---
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cntr, u, _, _, _, _, _ = fuzz.cmeans(
            data=X_values.T,
            c=self.n_clusters,
            m=self.fuzzy_m,
            error=self.fuzzy_error,
            maxiter=self.fuzzy_max_iter,
            init=None,
        )
        # u: shape (n_clusters, n_samples), same convention as Base_Model.py

        estimator_template = (
            DecisionTreeClassifier(random_state=self.random_state)
            if self.base_estimator is None
            else self.base_estimator
        )

        self.estimators_ = []

        for i in range(self.n_clusters):
            selected_indices = []

            for cls in self.classes_:
                class_indices = np.where(y_values == cls)[0]
                class_count = len(class_indices)
                if class_count == 0:
                    continue

                # Paper's selection rule: top `selection_percent` of this
                # class's patterns, ranked by membership to cluster i.
                n_select = round(self.selection_percent * class_count)

                # Floor added by us (see class docstring): never let a
                # class disappear entirely from a cluster's subset.
                n_select = max(n_select, min(self.min_samples_per_class, class_count))

                class_memberships = u[i, class_indices]
                # argsort ascending -> take the LAST n_select (highest membership)
                ranked_local = np.argsort(class_memberships)
                top_local = ranked_local[-n_select:] if n_select > 0 else np.array([], dtype=int)
                selected_indices.extend(class_indices[top_local])

            selected_indices = np.asarray(selected_indices, dtype=int)
            X_cluster = X_values[selected_indices]
            y_cluster = y_values[selected_indices]

            clf = clone(estimator_template)
            if "random_state" in clf.get_params():
                seed = self.random_state + i if self.random_state is not None else None
                clf.set_params(random_state=seed)
            clf.fit(X_cluster, y_cluster)
            self.estimators_.append(clf)

        return self

    def predict_proba(self, X):
        """
        Max-rule fusion (paper's Section 2): element-wise max, across all
        K classifiers, of each classifier's predict_proba output. Scores
        need not sum to 1 across classes -- see class docstring.
        """
        check_is_fitted(self, attributes=["classes_"])

        X_values = np.asarray(X)
        n_samples = X_values.shape[0]
        n_classes = len(self.classes_)

        max_probs = np.zeros((n_samples, n_classes))

        for clf in self.estimators_:
            local_probs = clf.predict_proba(X_values)
            local_classes = clf.classes_

            # Realign this classifier's columns to the GLOBAL class order
            # (robust fallback in case min_samples_per_class is set to 0
            # and a class genuinely never made it into some cluster).
            full_probs = np.zeros((n_samples, n_classes))
            for local_idx, cls in enumerate(local_classes):
                global_idx = np.where(self.classes_ == cls)[0][0]
                full_probs[:, global_idx] = local_probs[:, local_idx]

            max_probs = np.maximum(max_probs, full_probs)

        return max_probs

    def predict(self, X):
        """Class with the highest max-rule score (paper's Section 2)."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
