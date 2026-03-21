import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Imports opcionais

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn não disponível. Usando implementação própria de HMM (EM)")

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível.")

try:
    from scipy.stats import kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# 1. Features de regime

def build_regime_features(
    returns_df: pd.DataFrame,
    vol_window: int = 21,
    corr_window: int = 60,
) -> pd.DataFrame:
    """
    Constrói features para detecção de regime a partir dos retornos
    Inclui volatilidade, correlação média, skewness e drawdown
    """
    logger.info(f"construindo features de regime  T={len(returns_df)}, n={returns_df.shape[1]}")
    features = {}

    roll_vol = returns_df.rolling(vol_window).std()
    features["vol_mean"] = roll_vol.mean(axis=1)
    features["vol_max"]  = roll_vol.max(axis=1)
    features["vol_ratio"]  = roll_vol.max(axis=1) / (roll_vol.mean(axis=1) + 1e-8)
    features["port_return"]  = returns_df.mean(axis=1)
    features["port_return_ma"] = returns_df.mean(axis=1).rolling(vol_window).mean()

    roll_corr = []
    for i in range(returns_df.shape[0]):
        start = max(0, i - corr_window)
        window = returns_df.iloc[start:i+1]
        if len(window) >= 5:
            c = window.corr().values
            upper = c[np.triu_indices(c.shape[0], k=1)]
            roll_corr.append(float(np.mean(upper)))
        else:
            roll_corr.append(np.nan)
    features["avg_corr"] = roll_corr

    port = returns_df.mean(axis=1)
    features["port_skew"] = port.rolling(vol_window).skew()

    cumret = (1 + port).cumprod()
    rolling_max = cumret.rolling(corr_window, min_periods=1).max()
    features["drawdown"] = (cumret - rolling_max) / (rolling_max + 1e-8)

    df = pd.DataFrame(features, index=returns_df.index)
    logger.info(f"features construídas: {list(df.columns)}")
    return df


# Cópula Gaussiana simplificada (fallback)

class _GaussianCopulaSimple:
    """
    Cópula Gaussiana multivariada sem dependências externas
    Usada como fallback quando copula_class não é fornecida ou quando há dados insuficientes para ajustar a cópula principal
    """

    def __init__(self, U: Optional[np.ndarray] = None):
        self.R_ = None
        self.n_dim_ = None
        self.rho_mean_ = None
        if U is not None:
            self.fit(U)

    def fit(self, U: np.ndarray) -> "_GaussianCopulaSimple":
        T, d = U.shape
        self.n_dim_ = d
        Z = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
        R = np.corrcoef(Z.T)
        R = self._nearest_psd(R)
        self.R_ = R
        upper_idx = np.triu_indices(d, k=1)
        self.rho_mean_ = float(np.mean(R[upper_idx]))
        logger.debug(f"GaussianCopula ajustada  d={d}  rho_mean={self.rho_mean_:.3f}")
        return self

    def _nearest_psd(self, A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        A = (A + A.T) / 2
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, eps)
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(A_psd))
        A_psd = A_psd / np.outer(d, d)
        return A_psd

    def simulate(self, n_sim: int = 10000) -> np.ndarray:
        if self.R_ is None:
            raise RuntimeError("Cópula não ajustada")
        L = np.linalg.cholesky(self.R_)
        Z = np.random.standard_normal((n_sim, self.n_dim_))
        Z_corr = Z @ L.T
        U_sim = stats.norm.cdf(Z_corr)
        return U_sim.astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        return {"lower_tail": 0.0, "upper_tail": 0.0}


# regimeDetector

class RegimeDetector:
    """
    Detecta regimes latentes nos retornos via HMM ou GMM
    Exporta regime_labels compatível com RegimeAwareLSTMPredictor
    """

    REGIME_LABELS = {
        0: "low_vol",
        1: "medium_vol",
        2: "high_vol",
    }

    def __init__(
        self,
        n_regimes: int = 2,
        method: str = "hmm",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.method = method.lower()
        self.random_state = random_state
        self._model = None
        self._scaler = None
        self._labels = None
        self._proba = None
        self._features = None
        self._is_fitted = False

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        mask = features.isnull().any(axis=1)
        if mask.any():
            logger.info(f"removendo {mask.sum()} linhas com NaN das features")
        clean = features.dropna()
        self._features = clean

        X = clean.values.astype(np.float64)
        if SKLEARN_AVAILABLE:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        logger.info(
            f"ajustando RegimeDetector método={self.method}  "
            f"n_regimes={self.n_regimes}  T={len(X)}"
        )

        if self.method == "hmm":
            self._fit_hmm(X, clean.index)
        elif self.method == "gmm":
            self._fit_gmm(X, clean.index)
        else:
            raise ValueError(f"método desconhecido: {self.method}")

        self._is_fitted = True
        self._log_regime_summary()
        return self

    def _fit_hmm(self, X: np.ndarray, index: pd.Index):
        if HMMLEARN_AVAILABLE:
            model = hmmlearn_hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=self.random_state,
                tol=1e-4,
            )
            model.fit(X)
            labels = model.predict(X)
            proba = model.predict_proba(X)
            ll = model.score(X)
            logger.info(f"HMM (hmmlearn) ajustado  log-verossimilhança: {ll:.2f}")
            self._model = model
        else:
            labels, proba = self._fit_hmm_em(X)

        labels = self._order_regimes_by_vol(labels, X)
        self._labels = pd.Series(labels.astype(float), index=index)
        self._proba  = pd.DataFrame(proba, index=index,
                                    columns=[f"regime_{r}" for r in range(self.n_regimes)])

    def _fit_hmm_em(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, d = X.shape
        np.random.seed(self.random_state)
        means = X[np.random.choice(T, self.n_regimes, replace=False)]
        covs = [np.eye(d) for _ in range(self.n_regimes)]
        pi  = np.ones(self.n_regimes) / self.n_regimes
        A_mat = np.full((self.n_regimes, self.n_regimes),
                        1.0 / self.n_regimes)

        gamma = np.zeros((T, self.n_regimes))
        for iteration in range(50):
            for t in range(T):
                for k in range(self.n_regimes):
                    diff  = X[t] - means[k]
                    cov_k = covs[k] + np.eye(d) * 1e-6
                    sign, logdet = np.linalg.slogdet(cov_k)
                    exponent = -0.5 * diff @ np.linalg.solve(cov_k, diff)
                    gamma[t, k] = pi[k] * np.exp(exponent - 0.5 * logdet)
                row_sum = gamma[t].sum()
                if row_sum > 0:
                    gamma[t] /= row_sum
                else:
                    gamma[t] = 1.0 / self.n_regimes

            Nk = gamma.sum(axis=0)
            for k in range(self.n_regimes):
                if Nk[k] > 0:
                    means[k] = (gamma[:, k:k+1] * X).sum(axis=0) / Nk[k]
                    diff_mat = X - means[k]
                    covs[k]  = (gamma[:, k:k+1] * diff_mat).T @ diff_mat / Nk[k]
            pi    = Nk / T
            A_mat = A_mat  # simplificado

        labels = gamma.argmax(axis=1)
        return labels, gamma

    def _fit_gmm(self, X: np.ndarray, index: pd.Index):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn necessário para GMM.")
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200,
        )
        model.fit(X)
        labels = model.predict(X)
        proba  = model.predict_proba(X)
        labels = self._order_regimes_by_vol(labels, X)
        self._model  = model
        self._labels = pd.Series(labels.astype(float), index=index)
        self._proba  = pd.DataFrame(proba, index=index,
                                    columns=[f"regime_{r}" for r in range(self.n_regimes)])

    def _order_regimes_by_vol(self, labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        vol_col = 0
        means   = [X[labels == k, vol_col].mean() if (labels == k).any() else 0
                   for k in range(self.n_regimes)]
        order   = np.argsort(means)
        mapping = {old: new for new, old in enumerate(order)}
        return np.array([mapping[l] for l in labels])

    def _log_regime_summary(self):
        if self._labels is None:
            return
        counts = self._labels.dropna().astype(int).value_counts().sort_index()
        T = len(self._labels.dropna())
        for r, n in counts.items():
            label = self.REGIME_LABELS.get(r, f"regime_{r}")
            logger.info(f" Regime {r} ({label}): {n} dias ({n/T*100:.1f}%)")

    def predict(self) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._labels.copy()

    def predict_current(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return int(self._labels.dropna().iloc[-1])

    def predict_proba(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._proba.copy()

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        if HMMLEARN_AVAILABLE and isinstance(self._model, hmmlearn_hmm.GaussianHMM):
            return self._model.transmat_
        if self._labels is not None:
            labels = self._labels.dropna().astype(int).values
            A = np.zeros((self.n_regimes, self.n_regimes))
            for t in range(len(labels) - 1):
                A[labels[t], labels[t+1]] += 1
            row_sums = A.sum(axis=1, keepdims=True)
            return np.where(row_sums > 0, A / row_sums, 1.0 / self.n_regimes)
        return None

    def get_regime_persistence(self) -> Optional[pd.Series]:
        A = self.get_transition_matrix()
        if A is None:
            return None
        return pd.Series(
            np.diag(A),
            index=[f"regime_{r}" for r in range(self.n_regimes)],
            name="persistence",
        )


# CopulaPerRegime

def _try_import_copula():
    """Tenta importar GaussianCopula e StudentTCopula do pipeline"""
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    for candidate in [
        _here.parent / "copulas",
        _here.parent.parent / "src" / "copulas",
        _here / "copulas",
    ]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from copulas.elliptical import GaussianCopula, StudentTCopula
        return GaussianCopula, StudentTCopula
    except ImportError:
        pass
    try:
        from elliptical import GaussianCopula, StudentTCopula
        return GaussianCopula, StudentTCopula
    except ImportError:
        return None, None


class CopulaPerRegime:
    """
    Estima e armazena uma cópula separada para cada regime detectado.

    Estratégia:
        Se copula_class foi fornecida, usa ela
        Se n >= min_obs_rich: tenta t-Student 
        Se min_obs_per_regime <= n < min_obs_rich: usa GaussianCopula do pipeline
        Se n < min_obs_per_regime: fallback _GaussianCopulaSimple

    Isso garante que o regime de crise (maior n e maior rho) use uma cópula que capture tail dependence, em vez de sempre usar a Gaussiana simples
    """

    def __init__(
        self,
        copula_class=None,
        copula_kwargs: Optional[dict] = None,
        min_obs_per_regime: int = 60,
        min_obs_rich: int = 200,
    ):
        self.copula_class       = copula_class
        self.copula_kwargs      = copula_kwargs or {}
        self.min_obs_per_regime = min_obs_per_regime
        self.min_obs_rich       = min_obs_rich
        self.copulas_: Dict[int, object]  = {}
        self.n_obs_per_regime_: Dict[int, int] = {}
        self.n_regimes_: int = 0
        self._is_fitted = False

    def _fit_copula_for_regime(self, U_regime: np.ndarray, r: int) -> object:
        """
        Seleciona e ajusta a cópula mais adequada para o regime r
        Prioridade: copula_class fornecida > t-Student > Gaussiana pipeline > fallback
        """
        n = len(U_regime)

        if self.copula_class is not None:
            try:
                cop = self.copula_class(**self.copula_kwargs)
                cop.fit(U_regime)
                logger.info(f"Regime {r}: {self.copula_class.__name__} ajustada")
                return cop
            except Exception as e:
                logger.warning(f"Regime {r}: {self.copula_class.__name__} falhou ({e}). Usando fallback.")

        GaussianCopula, StudentTCopula = _try_import_copula()

        if n >= self.min_obs_rich and StudentTCopula is not None:
            try:
                cop = StudentTCopula(method="ifm")
                cop.fit(U_regime)
                logger.info(f"Regime {r}: StudentTCopula ajustada (tail dependence capturada)")
                return cop
            except Exception as e:
                logger.warning(f"Regime {r}: StudentTCopula falhou ({e}). Tentando Gaussiana.")

        if GaussianCopula is not None:
            try:
                cop = GaussianCopula(method="spearman")
                cop.fit(U_regime)
                logger.info(f"Regime {r}: GaussianCopula (pipeline) ajustada")
                return cop
            except Exception as e:
                logger.warning(f"Regime {r}: GaussianCopula falhou ({e}). Usando fallback")

        logger.info(f"Regime {r}: _GaussianCopulaSimple (fallback)")
        return _GaussianCopulaSimple(U_regime)

    def fit(
        self,
        uniformes: np.ndarray,
        regime_labels: pd.Series,
        index: Optional[pd.Index] = None,
    ) -> "CopulaPerRegime":
        if isinstance(uniformes, pd.DataFrame):
            index = uniformes.index
            U = uniformes.values
        else:
            U = uniformes.astype(np.float64)
            if index is None:
                index = pd.RangeIndex(len(U))

        if hasattr(regime_labels, "index") and index is not None:
            common = regime_labels.dropna().index.intersection(index)
            if len(common) < len(index) * 0.5:
                logger.warning("menos de 50% dos índices coincidem. Verificar alinhamento")
            labels_aligned = regime_labels.reindex(index)
        else:
            labels_aligned = pd.Series(regime_labels.values[:len(U)], index=index)

        regimes = sorted(labels_aligned.dropna().astype(int).unique())
        self.n_regimes_ = len(regimes)
        logger.info(f"ajustando cópulas por regime  {self.n_regimes_} regimes")

        for r in regimes:
            mask     = (labels_aligned == r).values
            U_regime = U[mask]
            n        = len(U_regime)
            self.n_obs_per_regime_[r] = n
            logger.info(f"  Regime {r}: {n} observações")

            if n < self.min_obs_per_regime:
                logger.warning(
                    f"  Regime {r}: {n} obs < mínimo ({self.min_obs_per_regime}) "
                    f"Usando Gaussian fallback."
                )
                self.copulas_[r] = _GaussianCopulaSimple(U_regime)
            else:
                self.copulas_[r] = self._fit_copula_for_regime(U_regime, r)

        self._is_fitted = True
        return self

    def simulate(self, regime: int, n_sim: int = 10000) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro")
        if regime not in self.copulas_:
            raise ValueError(f"Regime {regime} não encontrado. Disponíveis: {list(self.copulas_.keys())}")
        return self.copulas_[regime].simulate(n_sim)

    def get_regime_summary(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro")
        rows = []
        for r, cop in self.copulas_.items():
            td  = cop.tail_dependence() if hasattr(cop, "tail_dependence") else {}
            rho = getattr(cop, "rho_mean_", None)
            if rho is None and hasattr(cop, "R_") and cop.R_ is not None:
                d  = cop.R_.shape[0]
                idx = np.triu_indices(d, k=1)
                rho = float(np.mean(cop.R_[idx]))
            rows.append({
                "regime": r,
                "n_obs": self.n_obs_per_regime_.get(r, 0),
                "copula_type": type(cop).__name__,
                "lower_tail": td.get("lower_tail", 0.0),
                "upper_tail": td.get("upper_tail", 0.0),
                "rho_mean": rho,
            })
        return pd.DataFrame(rows).set_index("regime")


# 5. RegimeSwitchingCopula

class RegimeSwitchingCopula:
    """
    Constrói features de regime a partir dos retornos
    Detecta regimes via HMM ou GMM
    Ajusta cópulas condicionais por regime (t-Student quando possível)
    Exporta tudo no formato esperado pelo resto do pipeline
    """

    def __init__(
        self,
        n_regimes: int = 2,
        detector_method: str = "hmm",
        copula_class=None,
        copula_kwargs: Optional[dict] = None,
        vol_window: int = 21,
        corr_window: int = 60,
        min_obs_rich: int = 200,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.detector_method = detector_method
        self.copula_class  = copula_class
        self.copula_kwargs = copula_kwargs or {}
        self.vol_window  = vol_window
        self.corr_window = corr_window
        self.min_obs_rich = min_obs_rich
        self.random_state = random_state

        self.detector_ = RegimeDetector(
            n_regimes=n_regimes,
            method=detector_method,
            random_state=random_state,
        )
        self.copula_per_regime_ = CopulaPerRegime(
            copula_class=copula_class,
            copula_kwargs=copula_kwargs,
            min_obs_rich=min_obs_rich,
        )
        self.features_: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        returns_df: pd.DataFrame,
        uniformes: Optional[np.ndarray] = None,
        uniformes_index: Optional[pd.Index] = None,
    ) -> "RegimeSwitchingCopula":
        logger.info("RegimeSwitchingCopula.fit() iniciado")
        logger.info(f"  returns_df: {returns_df.shape}")
        if uniformes is not None:
            logger.info(f"  uniformes:  {uniformes.shape}")

        self.features_ = build_regime_features(
            returns_df,
            vol_window=self.vol_window,
            corr_window=self.corr_window,
        )

        self.detector_.fit(self.features_)
        regime_labels = self.detector_.predict()

        if uniformes is None:
            logger.warning(
                "uniformes não fornecidos, usando uniformes empíricos (rank-based) "
                "Para resultados corretos, passe uniformes de semi_parametric.py."
            )
            uniformes       = self._empirical_uniformes(returns_df)
            uniformes_index = returns_df.index
        else:
            if uniformes_index is None:
                uniformes_index = (returns_df.index if len(uniformes) == len(returns_df)
                                   else pd.RangeIndex(len(uniformes)))

        U_df = pd.DataFrame(
            uniformes,
            index=uniformes_index,
            columns=returns_df.columns[:uniformes.shape[1]],
        )
        self.copula_per_regime_.fit(U_df, regime_labels)

        self._is_fitted = True
        logger.info("RegimeSwitchingCopula ajustado com sucesso.")
        self._log_summary()
        return self

    def _empirical_uniformes(self, returns_df: pd.DataFrame) -> np.ndarray:
        T, d = returns_df.shape
        U = np.zeros((T, d))
        for j in range(d):
            col  = returns_df.iloc[:, j].values
            ranks = stats.rankdata(col, method="average")
            U[:, j] = ranks / (T + 1)
        return U.astype(np.float32)

    def _log_summary(self):
        logger.info("Resumo das cópulas por regime:")
        for r, cop in self.copula_per_regime_.copulas_.items():
            n = self.copula_per_regime_.n_obs_per_regime_.get(r, 0)
            logger.info(f"  Regime {r}: {type(cop).__name__}  {n} obs")
        A = self.get_transition_matrix()
        if A is not None:
            logger.info("Matriz de transição:")
            logger.info(f"\n{A.round(3)}")

    # Outputs para integração

    def get_regime_labels(self) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.detector_.predict()

    def get_current_regime(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.detector_.predict_current()

    def get_regime_probabilities(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.detector_.predict_proba()

    def simulate_current_regime(self, n_sim: int = 50000) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        regime = self.get_current_regime()
        logger.info(f"Simulando cópula do regime atual: {regime}")
        return self.copula_per_regime_.simulate(regime, n_sim)

    def simulate_regime(self, regime: int, n_sim: int = 50000) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.copula_per_regime_.simulate(regime, n_sim)

    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        A = self.detector_.get_transition_matrix()
        if A is None:
            return None
        return pd.DataFrame(
            A,
            index=[f"de_regime_{r}"  for r in range(self.n_regimes)],
            columns=[f"para_regime_{r}" for r in range(self.n_regimes)],
        )

    def get_persistence(self) -> Optional[pd.Series]:
        return self.detector_.get_regime_persistence()

    def get_summary(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.copula_per_regime_.get_regime_summary()

    def get_features(self) -> pd.DataFrame:
        return self.features_

    def compute_regime_statistics(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estatísticas dos retornos por regime"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        labels = self.get_regime_labels()
        common = returns_df.index.intersection(labels.dropna().index)
        port   = returns_df.loc[common].mean(axis=1)
        labs   = labels.loc[common].astype(int)
        rows   = []
        for r in sorted(labs.unique()):
            mask = labs == r
            ret  = port[mask]
            rows.append({
                "regime": r,
                "label": RegimeDetector.REGIME_LABELS.get(r, f"regime_{r}"),
                "n_dias": int(mask.sum()),
                "pct_total": float(mask.mean() * 100),
                "retorno_anual": float(ret.mean() * 252 * 100),
                "vol_anual": float(ret.std() * np.sqrt(252) * 100),
                "sharpe": float(ret.mean() / (ret.std() + 1e-8) * np.sqrt(252)),
                "skewness": float(ret.skew()),
                "excess_kurtosis": float(ret.kurtosis()),
                "max_drawdown": float(((1 + ret).cumprod() /
                                          (1 + ret).cumprod().cummax() - 1).min() * 100),
            })
        return pd.DataFrame(rows).set_index("regime")

    def save(self, path: Union[str, Path]):
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Modelo salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RegimeSwitchingCopula":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Modelo carregado de {path}")
        return obj


# Função de conveniência para main_pipeline.py

def detect_regimes_and_fit_copulas(
    returns_df: pd.DataFrame,
    uniformes: Optional[np.ndarray] = None,
    n_regimes: int = 2,
    detector_method: str = "hmm",
    copula_class=None,
    vol_window: int = 21,
    corr_window: int = 60,
    min_obs_rich: int = 200,
    save_path: Optional[str] = None,
) -> Tuple[RegimeSwitchingCopula, pd.Series, int]:
    """
    Wrapper
    """
    rsc = RegimeSwitchingCopula(
        n_regimes=n_regimes,
        detector_method=detector_method,
        copula_class=copula_class,
        vol_window=vol_window,
        corr_window=corr_window,
        min_obs_rich=min_obs_rich,
    )
    rsc.fit(returns_df, uniformes)

    regime_labels  = rsc.get_regime_labels()
    current_regime = rsc.get_current_regime()

    if save_path:
        rsc.save(save_path)

    return rsc, regime_labels, current_regime


# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    T       = 800
    n_ativos = 5
    dates   = pd.date_range("2019-01-01", periods=T, freq="B")

    regime_true = np.zeros(T, dtype=int)
    regime_true[200:350] = 1
    regime_true[550:650] = 1

    returns_sim = np.zeros((T, n_ativos))
    for t in range(T):
        if regime_true[t] == 0:
            cov = 0.3 * np.ones((n_ativos, n_ativos)) + 0.7 * np.eye(n_ativos)
            cov *= (0.01 ** 2)
        else:
            cov = 0.7 * np.ones((n_ativos, n_ativos)) + 0.3 * np.eye(n_ativos)
            cov *= (0.025 ** 2)
            returns_sim[t] -= 0.001
        L = np.linalg.cholesky(cov + np.eye(n_ativos) * 1e-8)
        returns_sim[t] += L @ np.random.standard_normal(n_ativos)

    tickers    = [f"ATIVO{i+1}" for i in range(n_ativos)]
    returns_df = pd.DataFrame(returns_sim, index=dates, columns=tickers)

    print(f"returns_df: {returns_df.shape}")
    print(f"Regime 0 (calmo): {(regime_true==0).sum()} dias")
    print(f"Regime 1 (crise): {(regime_true==1).sum()} dias")

    # Construindo features

    print("\n Construindo features de regime ")
    features = build_regime_features(returns_df, vol_window=21, corr_window=60)
    print(f"features: {features.shape}  colunas: {list(features.columns)}")

    # RegimeDetector HMM

    print("\n RegimeDetector (HMM) ")
    detector = RegimeDetector(n_regimes=2, method="hmm")
    detector.fit(features)
    labels = detector.predict()
    print(f"regime_labels: {labels.value_counts().to_dict()}")
    print(f"Regime atual: {detector.predict_current()}")

    valid = labels.dropna().astype(int)
    acc   = max(
        np.mean(valid.values == regime_true[len(regime_true)-len(valid):]),
        np.mean(valid.values != regime_true[len(regime_true)-len(valid):])
    )
    print(f"Acurácia (melhor alinhamento): {acc:.1%}")

    # Uniformes empíricos

    U = np.zeros((T, n_ativos))
    for j in range(n_ativos):
        ranks  = stats.rankdata(returns_sim[:, j])
        U[:, j] = ranks / (T + 1)

    # RegimeSwitchingCopula completo

    print("\n RegimeSwitchingCopula.fit() ")
    rsc, regime_labels, current_regime = detect_regimes_and_fit_copulas(
        returns_df=returns_df,
        uniformes=U,
        n_regimes=2,
        detector_method="hmm",
    )

    print(f"\nRegime atual: {current_regime}")
    print(f"\nResumo por regime:\n{rsc.get_summary()}")

    stats_by_regime = rsc.compute_regime_statistics(returns_df)
    print(f"\nEstatísticas por regime:\n{stats_by_regime.round(3)}")

    A = rsc.get_transition_matrix()
    if A is not None:
        print(f"\nMatriz de transição:\n{A.round(3)}")

    p = rsc.get_persistence()
    if p is not None:
        print(f"\nPersistência dos regimes:\n{p.round(3)}")

    # Simulação

    print("\n Simulando cópula do regime atual ")
    U_sim = rsc.simulate_current_regime(n_sim=1000)
    print(f"U_sim: {U_sim.shape}  min={U_sim.min():.3f}  max={U_sim.max():.3f}")

    # Integração com RegimeAwareLSTMPredictor

    print("\n Integração: formato para RegimeAwareLSTMPredictor ")
    rl = rsc.get_regime_labels()
    print(f"regime_labels dtype={rl.dtype}, shape={rl.shape}")
    print(f"Pronto para: ra_lstm.fit(params_df, regime_labels=rl)")
    print(f"Pronto para: ra_lstm.predict(current_regime={current_regime})")

    print("\n Todos os testes concluídos.")