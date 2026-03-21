import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union, Callable
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Transformação PIT 

class PITTransformer:
    """
    Transforma resíduos/retornos em pseudo-observações uniformes [0,1]

    Três métodos disponíveis:
        'empirical': F_n(x) = rank(x)/(T+1) não-paramétrico CML
        'normal': Φ(x/σ),assume normalidade 
        'kernel': KDE + CDF, semi-paramétrico
    """

    def __init__(self, method: str = "empirical"):
        self.method = method
        self._fitted_params: Dict[str, dict] = {}
        self._is_fitted = False

    def fit_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        column_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Ajusta e transforma dados para uniformes [0,1]
        """
        if isinstance(data, pd.DataFrame):
            column_names = column_names or list(data.columns)
            X = data.values.astype(np.float64)
        else:
            X = data.astype(np.float64)

        T, d = X.shape
        U = np.zeros((T, d))

        for j in range(d):
            col = X[:, j]
            name = column_names[j] if column_names else f"col_{j}"
            U[:, j], params = self._transform_single(col)
            self._fitted_params[name] = params

        # Clip para evitar 0 e 1 exatos
        U = np.clip(U, 1e-9, 1-1e-9)
        self._is_fitted = True

        logger.info(
            f"PIT ({self.method}): {X.shape} → U {U.shape}  "
            f"range [{U.min():.4f}, {U.max():.4f}]"
        )
        return U.astype(np.float32)

    def _transform_single(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Transforma uma única coluna para [0,1]"""
        if self.method == "empirical":
            ranks = rankdata(x, method="average")
            u = ranks / (len(x) + 1)
            return u, {"method": "empirical"}

        elif self.method == "normal":
            mu = np.mean(x)
            sigma = np.std(x, ddof=1)
            u = stats.norm.cdf(x, loc=mu, scale=sigma)
            return u, {"method": "normal", "mu": mu, "sigma": sigma}

        elif self.method == "kernel":
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(x, bw_method="silverman")
            # CDF via integração numérica
            x_sorted = np.sort(x)
            cdf_vals = np.zeros(len(x))
            for i, xi in enumerate(x):
                cdf_vals[i] = float(kde.integrate_box_1d(-np.inf, xi))
            # Mapear de volta à ordem original
            sort_idx = np.argsort(x)
            u = np.empty(len(x))
            u[sort_idx] = cdf_vals
            return u, {"method": "kernel"}

        else:
            raise ValueError(f"método desconhecido: {self.method}")

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Transforma novos dados usando os parâmetros ajustados.
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_transform() primeiro.")

        if isinstance(data, pd.DataFrame):
            X = data.values.astype(np.float64)
        else:
            X = data.astype(np.float64)

        T, d = X.shape
        U = np.zeros((T, d))
        names = list(self._fitted_params.keys())

        for j in range(d):
            name = names[j] if j < len(names) else f"col_{j}"
            params = self._fitted_params.get(name, {"method": "empirical"})

            if params.get("method") == "normal":
                U[:, j] = stats.norm.cdf(
                    X[:, j], loc=params["mu"], scale=params["sigma"]
                )
            else:
                # Fallback: empirical
                ranks = rankdata(X[:, j], method="average")
                U[:, j] = ranks / (len(X[:, j]) + 1)

        return np.clip(U, 1e-9, 1-1e-9).astype(np.float32)


# 2. Estimador IFM

class IFMEstimator:
    """
    Estima cada marginal separadamente 
    Estima a cópula com as uniformes obtidas das marginais
    """

    def __init__(
        self,
        marginal_dist: str = "t",
        copula_factory: Optional[Callable] = None,
    ):
        
        self.marginal_dist = marginal_dist
        self.copula_factory = copula_factory
        self.marginal_params_: Dict[str, dict] = {}
        self.U_: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit_marginals(
        self, returns_df: pd.DataFrame
    ) -> np.ndarray:
        T, d = returns_df.shape
        U = np.zeros((T, d))

        for j, col in enumerate(returns_df.columns):
            x = returns_df[col].dropna().values

            if self.marginal_dist == "t":
                nu, loc, scale = stats.t.fit(x)
                U[:len(x), j] = stats.t.cdf(x, df=nu, loc=loc, scale=scale)
                self.marginal_params_[col] = {"dist": "t", "nu": nu, "loc": loc, "scale": scale}

            elif self.marginal_dist == "normal":
                mu, sigma = stats.norm.fit(x)
                U[:len(x), j] = stats.norm.cdf(x, loc=mu, scale=sigma)
                self.marginal_params_[col] = {"dist": "normal", "mu": mu, "sigma": sigma}

            else:
                # Fallback: empírico
                ranks = rankdata(x, method="average")
                U[:len(x), j] = ranks / (len(x) + 1)
                self.marginal_params_[col] = {"dist": "empirical"}

        U = np.clip(U, 1e-9, 1-1e-9)
        self.U_ = U.astype(np.float32)
        logger.info(f"IFM marginais ajustadas: {d} ativos  dist={self.marginal_dist}")
        return self.U_

    def fit_copula(self, copula_instance) -> object:
        if self.U_ is None:
            raise RuntimeError("Execute fit_marginals() primeiro.")
        copula_instance.fit(self.U_)
        self._is_fitted = True
        return copula_instance

# Estimador CML

class CMLEstimator:
    """

    Usa ranks empíricos como estimativas não-paramétricas das marginais.
    """

    def __init__(self):
        self.U_: Optional[np.ndarray] = None
        self._pit = PITTransformer(method="empirical")

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Transforma dados para pseudo-observações via ranks
        """
        self.U_ = self._pit.fit_transform(data)
        return self.U_

    def fit_and_transform(
        self,
        returns_df: pd.DataFrame,
        copula_instance,
    ) -> Tuple[np.ndarray, object]:
        U = self.transform(returns_df)
        copula_instance.fit(U)
        return U, copula_instance


# Estimação de Correlação Robusta

class RobustCorrelationEstimator:
    """
    Estimadores robustos de correlação para a matriz de cópulas

    Métodos disponíveis:
        'pearson': correlação linear padrão
        'spearman': correlação de postos 
        'kendall': tau de Kendall 
        'mcd': Minimum Covariance Determinant 
    """

    @staticmethod
    def estimate(
        U: np.ndarray,
        method: str = "spearman",
    ) -> np.ndarray:
        """
        Estima matriz de correlação de pseudo-observações

        Parametros: 
        U: np.ndarray (T, d), pseudo-observações
        method : str

        Retornos:
        R: np.ndarray (d, d), matriz de correlação PSD
        """
        T, d = U.shape

        if method == "pearson":
            R = np.corrcoef(U.T)

        elif method == "spearman":
            # Correlação de Spearman das uniformes
            df = pd.DataFrame(U)
            R = df.corr(method="spearman").values
            # Converter para correlação da cópula Gaussiana
            R = np.sin(np.pi / 6 * R) * 2
            np.fill_diagonal(R, 1.0)

        elif method == "kendall":
            R = np.eye(d)
            for i in range(d):
                for j in range(i+1, d):
                    tau, _ = stats.kendalltau(U[:, i], U[:, j])
                    rho = np.sin(np.pi * tau / 2)
                    R[i, j] = rho
                    R[j, i] = rho

        elif method == "mcd":
            try:
                from sklearn.covariance import MinCovDet
                # Transformar para normal primeiro
                Z = stats.norm.ppf(np.clip(U, 1e-6, 1-1e-6))
                mcd = MinCovDet(random_state=42).fit(Z)
                cov = mcd.covariance_
                std = np.sqrt(np.diag(cov))
                R = cov / np.outer(std, std)
            except ImportError:
                logger.warning("MCD requer sklearn. Usando Spearman.")
                return RobustCorrelationEstimator.estimate(U, method="spearman")

        else:
            raise ValueError(f"método desconhecido: {method}")

        # Garantir PSD
        try:
            from copulas.elliptical import nearest_psd
        except ImportError:
            try:
                from elliptical import nearest_psd
            except ImportError:
                # Fallback inline
                def nearest_psd(A, eps=1e-6):
                    A = (A + A.T) / 2
                    eigvals, eigvecs = np.linalg.eigh(A)
                    eigvals = np.maximum(eigvals, eps)
                    A2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    d2 = np.maximum(np.sqrt(np.diag(A2)), 1e-10)
                    return A2 / np.outer(d2, d2)
        return nearest_psd(R)

    @staticmethod
    def shrinkage_correlation(
        U: np.ndarray,
        shrinkage: float = 0.1,
    ) -> np.ndarray:
        """
        Correlação amostral com shrinkage para o alvo diagonal 
        Reduz erro de estimação em d grande.

        R_shrunk = (1-δ) * R_sample + δ * I
        """
        R = np.corrcoef(U.T)
        d = R.shape[0]
        return (1 - shrinkage) * R + shrinkage * np.eye(d)


# Estimação de cópula multivariada completa

def estimate_copula(
    data: Union[pd.DataFrame, np.ndarray],
    copula_type: str = "gaussian",
    pit_method: str = "empirical",
    estimation_method: str = "cml",
    **copula_kwargs,
) -> Tuple[np.ndarray, object]:

    # PIT → uniformes
    pit = PITTransformer(method=pit_method)
    U = pit.fit_transform(data)

    # Instanciar cópula
    copula = _instantiate_copula(copula_type, **copula_kwargs)

    # Ajustar
    if estimation_method in ("cml", "mle"):
        copula.fit(U)
    elif estimation_method == "ifm":
        if isinstance(data, pd.DataFrame):
            ifm = IFMEstimator(marginal_dist="t")
            U = ifm.fit_marginals(data)
        copula.fit(U)
    else:
        raise ValueError(f"método desconhecido: {estimation_method}")

    logger.info(
        f"Cópula {copula_type} ajustada  método={estimation_method}  "
        f"AIC={getattr(copula, 'aic_', np.nan):.2f}"
    )
    return U, copula


def _instantiate_copula(copula_type: str, **kwargs) -> object:
    """Instancia cópula pelo nome."""
    try:
        from copulas.elliptical import GaussianCopula, StudentTCopula
        from copulas.archimedean import ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula
    except ImportError:
        from elliptical import GaussianCopula, StudentTCopula
        from archimedean import ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula

    registry = {
        "gaussian": GaussianCopula,
        "t":        StudentTCopula,
        "student":  StudentTCopula,
        "clayton":  ClaytonCopula,
        "gumbel":   GumbelCopula,
        "frank":    FrankCopula,
        "joe":      JoeCopula,
    }
    cls = registry.get(copula_type.lower())
    if cls is None:
        raise ValueError(f"Tipo desconhecido: {copula_type}. Opções: {list(registry.keys())}")
    return cls(**kwargs)


#  Diagnósticos de Estimação

def check_uniform_quality(U: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """
    Verifica se as pseudo-observações são realmente uniformes
    Aplica teste KS e AD para cada coluna

    """
    T, d = U.shape
    rows = []

    for j in range(d):
        u_col = U[:, j]

        # KS test vs Uniforme(0,1)
        ks_stat, ks_pval = stats.kstest(u_col, "uniform")

        # Anderson-Darling
        try:
            ad_result = stats.anderson(u_col, dist="uniform")
            ad_stat = ad_result.statistic
        except Exception:
            ad_stat = np.nan

        rows.append({
            "column": j,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "ad_statistic": ad_stat,
            "is_uniform": ks_pval > alpha,
            "mean": float(np.mean(u_col)),
            "std": float(np.std(u_col)),
        })

    df = pd.DataFrame(rows)
    n_ok = df["is_uniform"].sum()
    logger.info(f"Qualidade das uniformes: {n_ok}/{d} passaram no KS test (α={alpha})")
    return df

#  Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    T, d = 800, 5
    dates = pd.date_range("2020-01-01", periods=T, freq="B")

    # Dados com correlacao conhecida 
    R_true = np.array([
        [1.0, 0.6, 0.6, 0.2, 0.2],
        [0.6, 1.0, 0.6, 0.2, 0.2],
        [0.6, 0.6, 1.0, 0.2, 0.2],
        [0.2, 0.2, 0.2, 1.0, 0.6],
        [0.2, 0.2, 0.2, 0.6, 1.0],
    ])
    L_true = np.linalg.cholesky(R_true)
    Z = np.random.standard_normal((T, d)) @ L_true.T
    chi2 = np.random.chisquare(5, T)
    returns_sim = (Z / np.sqrt(chi2[:, None] / 5)) * 0.012
    returns_df = pd.DataFrame(returns_sim, index=dates,
                              columns=[f"ATIVO{i+1}" for i in range(d)])

    rho_true_mean = R_true[np.triu_indices(d, k=1)].mean()
    print(f"returns_df: {returns_df.shape}")
    print(f"Correlacao verdadeira, media off-diag: {rho_true_mean:.3f}")

    #  PIT Empirico 
    print("\n PITTransformer (empirical) ")
    pit = PITTransformer(method="empirical")
    U = pit.fit_transform(returns_df)
    print(f"U: {U.shape}  range [{U.min():.4f}, {U.max():.4f}]")
    quality = check_uniform_quality(U)
    print(f"KS test: {quality['is_uniform'].sum()}/{d} uniformes ok")

    #  PIT Normal 
    print("\n PITTransformer (normal) ")
    pit_n = PITTransformer(method="normal")
    U_n = pit_n.fit_transform(returns_df)
    print(f"U_normal: {U_n.shape}")

    #  CML 
    print("\n CMLEstimator ")
    cml = CMLEstimator()
    U_cml = cml.transform(returns_df)
    print(f"U_cml: {U_cml.shape}")

    #  Correlacao Robusta 
    print("\n RobustCorrelationEstimator ")
    for method in ["pearson", "spearman", "kendall"]:
        R = RobustCorrelationEstimator.estimate(U, method=method)
        upper = R[np.triu_indices(d, k=1)]
        print(f"  {method:10s}: rho_mean={upper.mean():.4f}  min={upper.min():.4f}  max={upper.max():.4f} (verdadeiro: {rho_true_mean:.3f})")

    #  IFM 
    print("\n IFMEstimator ")
    ifm = IFMEstimator(marginal_dist="t")
    U_ifm = ifm.fit_marginals(returns_df)
    print(f"U_ifm: {U_ifm.shape}")
    quality_ifm = check_uniform_quality(U_ifm)
    print(f"KS test IFM: {quality_ifm['is_uniform'].sum()}/{d} uniformes ok")

    # --
    # INTEGRACAO COM elliptical.py
    # --
    print("\n" + "=" * 60)
    print("INTEGRACAO: estimation.py + elliptical.py")
    print("=" * 60)

    try:
        from copulas.elliptical import (
            GaussianCopula, StudentTCopula, empirical_tail_dependence,
        )
        ELLIPTICAL_AVAILABLE = True
    except ImportError:
        try:
            from elliptical import (
                GaussianCopula, StudentTCopula, empirical_tail_dependence,
            )
            ELLIPTICAL_AVAILABLE = True
        except ImportError:
            ELLIPTICAL_AVAILABLE = False
            print("  elliptical.py nao encontrado — adicione src/ ao PYTHONPATH")

    if ELLIPTICAL_AVAILABLE:
        # PIT -> GaussianCopula
        print("\n PIT (CML) -> GaussianCopula ")
        gc = GaussianCopula(method="spearman")
        gc.fit(U_cml)
        rho_est = gc._rho_mean()
        rho_err = abs(rho_est - rho_true_mean)
        print(f"rho_mean estimado: {rho_est:.4f}")
        print(f"rho_mean verdadeiro: {rho_true_mean:.4f}")
        print(f"erro absoluto: {rho_err:.4f} ({'OK' if rho_err < 0.05 else 'ALTO'})")
        R_est    = gc.R_
        rmse_R   = float(np.sqrt(np.mean((R_true[np.triu_indices(d,k=1)] - R_est[np.triu_indices(d,k=1)])**2)))
        print(f"  RMSE correlacoes   : {rmse_R:.4f} ({'OK' if rmse_R < 0.08 else 'ALTO'})")
        print(f"  AIC={gc.aic_:.2f}  LL={gc.log_likelihood_:.2f}")

        # PIT -> StudentTCopula
        print("\n PIT (CML) -> StudentTCopula ")
        tc = StudentTCopula(method="ifm")
        tc.fit(U_cml)
        print(f"nu estimado: {tc.nu_:.2f}")
        print(f"AIC t={tc.aic_:.2f} vs AIC Gaussiana={gc.aic_:.2f}")
        winner = "t-Student" if tc.aic_ < gc.aic_ else "Gaussiana"
        print(f"Selecao por AIC: {winner} (esperado: t-Student, dados sao t nu=5)")
        td_emp = empirical_tail_dependence(U_cml, alpha=0.10)
        td_ana = tc.tail_dependence()
        print(f"Tail dep empirica  : lL={td_emp['lower_tail']:.3f} lU={td_emp['upper_tail']:.3f}")
        print(f"Tail dep analitica : lL={td_ana['lower_tail']:.3f} (nu pode estar superestimado)")

        # Simulate -> check uniformidade
        print("\n Simulate -> uniformidade ")
        U_sim_gc = gc.simulate(500, seed=42)
        U_sim_tc = tc.simulate(500, seed=42)
        q_gc = check_uniform_quality(U_sim_gc)
        q_tc = check_uniform_quality(U_sim_tc)
        print(f"  Gaussiana: {q_gc['is_uniform'].sum()}/{d} uniformes ok")
        print(f"  t-Student: {q_tc['is_uniform'].sum()}/{d} uniformes ok")

        # estimate_copula() one-liner
        print("\n estimate_copula() one-liner ")
        U_ol, cop_ol = estimate_copula(returns_df, copula_type="gaussian", pit_method="empirical")
        print(f"U: {U_ol.shape}  {type(cop_ol).__name__}  AIC={cop_ol.aic_:.2f}")

    print("Teste concluído.")
