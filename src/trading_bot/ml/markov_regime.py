"""
Markov Regime Detector

Hidden Markov Model (HMM) for market regime detection.
Used by the Hybrid HMM + 5% Stop Loss strategy.

Supports model persistence (save/load) for consistent regime labels.

Uses forward-only filtering (not Viterbi) for walk-forward predictions
to avoid look-ahead bias. Viterbi is a global optimizer that assigns
states using future data within each prediction window, inflating
regime detection accuracy in backtests.
"""

import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
from hmmlearn import hmm
from hmmlearn import _hmmc

logger = logging.getLogger(__name__)


class MarkovRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Detects 4 market regimes: Strong Bull, Weak Bull, Weak Bear, and Strong Bear.
    Uses walk-forward training to avoid look-ahead bias.

    Supports model persistence to maintain consistent regime labels across runs.
    """

    # Fixed regime labels for 4-state HMM
    # State numbers are mapped to these labels based on return characteristics
    REGIME_LABELS = {
        0: "state_0",
        1: "state_1",
        2: "state_2",
        3: "state_3",
    }

    def __init__(
        self,
        n_states: int = 4,
        min_training_days: int = 63,
        retrain_frequency: int = 21,
        random_state: int = 42
    ):
        self.n_states = n_states
        self.min_training_days = min_training_days
        self.retrain_frequency = retrain_frequency
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self._fitted = False
        self._state_labels: Dict[int, str] = {}  # State number -> label mapping
        self._state_means: Dict[int, float] = {}  # State number -> mean return

    def fit(self, data: pd.DataFrame) -> "MarkovRegimeDetector":
        """Fit HMM to data with convergence retries."""
        returns = data['close'].pct_change().dropna().values.reshape(-1, 1)

        if len(returns) < self.min_training_days:
            raise ValueError(f"Need at least {self.min_training_days} days of data")

        # Try multiple random seeds for convergence
        for attempt in range(5):
            try:
                self.model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=2000,
                    random_state=self.random_state + attempt,
                    init_params="stmc",
                    params="stmc",
                    tol=1e-4,
                )
                self.model.fit(returns)
                # Verify the model has valid parameters
                if not np.isnan(self.model.means_).any() and not np.isnan(self.model.transmat_).any():
                    self._fitted = True
                    return self
                logger.warning(f"Fit attempt {attempt+1} produced NaN parameters, retrying")
            except Exception as e:
                logger.warning(f"Fit attempt {attempt+1} failed: {e}")

        # Last resort: fit with default params
        logger.warning("All fit attempts failed, using fallback configuration")
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=self.random_state,
            init_params="stmc",
            params="stmc",
            tol=1e-2,
        )
        self.model.fit(returns)
        self._fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict regimes for all data points."""
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        returns = data['close'].pct_change().dropna().values.reshape(-1, 1)
        regimes = self.model.predict(returns)

        # Align index with original data
        index = data.index[1:]  # Skip first due to pct_change()
        return pd.Series(regimes, index=index, name='regime')

    def predict_walkforward(self, data: pd.DataFrame) -> tuple[pd.Series, Dict[int, Dict[int, float]]]:
        """
        Predict regimes using walk-forward training with forward-only filtering.

        Retrains model every retrain_frequency days to avoid look-ahead bias.
        Uses forward-only filtering (not Viterbi) for regime assignment,
        which prevents look-ahead bias within each prediction window.

        Viterbi decoding finds the globally optimal state sequence across the
        entire prediction window, meaning the state for day 1 of a 21-day window
        considers data from days 2-21. Forward-only filtering computes
        P(state_t | observations_1..t) using only past observations, which is
        what a real-time trader would have access to.

        Returns:
            tuple: (regimes Series, dict mapping date_index -> state_means for that period)
        """
        returns = data['close'].pct_change().dropna()
        regimes = pd.Series(index=returns.index, dtype=int)
        state_means_by_period = {}

        last_model = None  # Track last successful model for fallback

        # Initial training
        train_end = self.min_training_days
        while train_end < len(returns):
            # Train on historical data only (expanding window)
            train_data = returns.iloc[:train_end].values.reshape(-1, 1)

            model = None
            fit_success = False

            # Try multiple random seeds for convergence
            for attempt in range(3):
                try:
                    candidate = hmm.GaussianHMM(
                        n_components=self.n_states,
                        covariance_type="diag",
                        n_iter=1000,
                        random_state=self.random_state + attempt,
                        init_params="stmc",
                        params="stmc",
                        tol=1e-4,
                    )
                    candidate.fit(train_data)

                    # Verify the model converged to valid parameters
                    if (not np.isnan(candidate.means_).any() and
                        not np.isnan(candidate.transmat_).any() and
                        np.all(candidate.transmat_.sum(axis=1) > 0.99)):
                        model = candidate
                        fit_success = True
                        break
                except Exception as e:
                    logger.debug(f"HMM attempt {attempt+1} failed for window {train_end}: {e}")

            if fit_success and model is not None:
                last_model = model

                # Store state means for this period's model
                if hasattr(model, 'means_'):
                    state_means = model.means_.flatten()
                    state_means_dict = {i: state_means[i] for i in range(len(state_means))}
                    state_means_by_period[train_end] = state_means_dict

                # Predict for next period using forward-only filtering
                predict_start = train_end
                predict_end = min(train_end + self.retrain_frequency, len(returns))

                if predict_start < predict_end:
                    predict_data = returns.iloc[predict_start:predict_end].values.reshape(-1, 1)
                    regimes.iloc[predict_start:predict_end] = self._predict_forward_only(
                        model, predict_data
                    )

            elif last_model is not None:
                # Fallback: reuse last successful model for this window
                logger.debug(f"Reusing last model for window ending at {train_end}")
                try:
                    predict_start = train_end
                    predict_end = min(train_end + self.retrain_frequency, len(returns))
                    if predict_start < predict_end:
                        predict_data = returns.iloc[predict_start:predict_end].values.reshape(-1, 1)
                        regimes.iloc[predict_start:predict_end] = self._predict_forward_only(
                            last_model, predict_data
                        )
                except Exception:
                    logger.warning(f"All models failed for window ending at {train_end}")
            else:
                logger.warning(f"No model available for window ending at {train_end}")

            train_end += self.retrain_frequency

        return regimes, state_means_by_period

    def _predict_forward_only(self, model, observations: np.ndarray) -> np.ndarray:
        """
        Predict regimes using forward-only filtering (no look-ahead).

        Uses the forward algorithm to compute P(state_t | obs_1, ..., obs_t),
        which only uses past observations. This avoids the look-ahead bias
        present in Viterbi decoding (which uses the full observation sequence).

        Args:
            model: A fitted GaussianHMM model
            observations: Array of shape (n_samples, 1) to predict

        Returns:
            Array of state labels (one per observation)
        """
        try:
            # Compute log emission probabilities for each observation
            log_frame_prob = model._compute_log_likelihood(observations)

            # Run the forward algorithm only (no backward pass)
            # This gives log P(state_t | obs_1, ..., obs_t) for each t
            log_prob, fwdlattice = _hmmc.forward_log(
                model.startprob_,
                model.transmat_,
                log_frame_prob
            )

            # Forward-only state assignment: argmax of forward probabilities
            # This uses only observations up to time t (no future data)
            return fwdlattice.argmax(axis=1)

        except Exception as e:
            # CRITICAL: Viterbi uses FUTURE data - introduces look-ahead bias!
            # If forward-only fails, we MUST use an alternative that doesn't use future data
            logger.error(f"Forward-only filtering failed ({e}), using safe fallback (no Viterbi!)")

            # Safe fallback: Use only emission probabilities for each time step
            # This picks the most likely state at each time independently (no temporal smoothing)
            # It's suboptimal but has NO look-ahead bias
            try:
                log_frame_prob = model._compute_log_likelihood(observations)
                # argmax of emissions = most likely state at each t using only that observation
                return log_frame_prob.argmax(axis=1)
            except Exception as e2:
                logger.error(f"All filtering failed ({e2}), using state 0 fallback")
                # Last resort: default to state 0 (conservative, no bias)
                return np.zeros(len(observations), dtype=int)

    def _label_regimes(
        self,
        regimes_numeric: pd.Series,
        benchmark_data: pd.DataFrame
    ) -> Dict[int, str]:
        """
        Create consistent state labels based on HMM state means.

        Returns dict mapping state number to label:
        {0: 'strong_bull', 1: 'weak_bull', 2: 'weak_bear', 3: 'strong_bear'}

        Labels are assigned by ranking states by their mean returns.
        """
        returns = benchmark_data['close'].pct_change().dropna()
        returns = returns.reindex(regimes_numeric.index).fillna(0)

        # Calculate mean return for each state
        stats = {}
        valid_regimes = regimes_numeric.dropna()
        valid_returns = returns.loc[valid_regimes.index]

        for state in range(self.n_states):
            mask = valid_regimes == state
            if mask.sum() > 0:
                stats[state] = valid_returns.loc[mask].mean()
            else:
                stats[state] = 0.0

        # Rank states by return and assign labels
        sorted_states = sorted(stats.items(), key=lambda x: x[1], reverse=True)

        labels = {}
        if len(sorted_states) == 4:
            labels[sorted_states[0][0]] = 'strong_bull'
            labels[sorted_states[1][0]] = 'weak_bull'
            labels[sorted_states[2][0]] = 'weak_bear'
            labels[sorted_states[3][0]] = 'strong_bear'
        elif len(sorted_states) == 3:
            labels[sorted_states[0][0]] = 'bull'
            labels[sorted_states[-1][0]] = 'bear'
            labels[sorted_states[1][0]] = 'neutral'
        else:
            labels[sorted_states[0][0]] = 'neutral'

        return labels

    def save(self, path: str) -> bool:
        """
        Save HMM model and state labels to file.

        Args:
            path: Path to save the model (e.g., "data/hmm_model.pkl")

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._fitted or self.model is None:
            return False

        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'model': self.model,
                'state_labels': self._state_labels,
                'state_means': self._state_means,
                'n_states': self.n_states,
                'random_state': self.random_state,
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            return True
        except Exception as e:
            logger.error(f"Failed to save HMM model: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load HMM model and state labels from file.

        Args:
            path: Path to load the model from (e.g., "data/hmm_model.pkl")

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            save_path = Path(path)
            if not save_path.exists():
                return False

            with open(save_path, 'rb') as f:
                data = pickle.load(f)

            self.model = data.get('model')
            self._state_labels = data.get('state_labels', {})
            self._state_means = data.get('state_means', {})
            self.n_states = data.get('n_states', 4)
            self.random_state = data.get('random_state', 42)
            self._fitted = self.model is not None

            return True
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            return False


