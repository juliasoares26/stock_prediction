import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, Union
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Estimador empirico de tail dependence

def empirical_tail_dependence(
    U: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Estimador empirico nao-parametrico de tail dependence CFG

    Fundamento: lambda_L = P(U <= alpha  V <= alpha) / alpha
                lambda_U = P(U >= 1-alpha  V >= 1-alpha) / alpha

    """
    if isinstance(U, pd.DataFrame):
        U = U.values
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        raise ValueError("U deve ter pelo menos 2 colunas")

    T, d = U.shape
    k = max(1, int(T * alpha))

    if d == 2:
        # Caso bivariado direto
        from scipy.stats import rankdata
        r0 = rankdata(U[:, 0])
        r1 = rankdata(U[:, 1])
        lam_L = float(np.mean((r0 <= k) & (r1 <= k))) / (k / T)
        lam_U = float(np.mean((r0 >= T - k) & (r1 >= T - k))) / (k / T)
        return {
            "lower_tail": float(np.clip(lam_L, 0, 1)),
            "upper_tail": float(np.clip(lam_U, 0, 1)),
            "alpha": alpha,
            "n_pairs": 1,
        }
    else:
        # Media sobre todos os pares
        from scipy.stats import rankdata
        lams_L, lams_U = [], []
        for i in range(d):
            for j in range(i + 1, d):
                ri = rankdata(U[:, i])
                rj = rankdata(U[:, j])
                lams_L.append(float(np.mean((ri <= k) & (rj <= k))) / (k / T))
                lams_U.append(float(np.mean((ri >= T - k) & (rj >= T - k))) / (k / T))
        return {
            "lower_tail": float(np.clip(np.mean(lams_L), 0, 1)),
            "upper_tail": float(np.clip(np.mean(lams_U), 0, 1)),
            "alpha": alpha,
            "n_pairs": len(lams_L),
        }


def analytical_tail_dependence_t(nu: float, rho: float) -> Dict[str, float]:
    """
    Tail dependence analitica da copula t-Student.
    lambda_L = lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
    """
    rho = np.clip(rho, -0.999, 0.999)
    arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lam = float(2 * stats.t.cdf(arg, df=nu + 1))
    return {"lower_tail": lam, "upper_tail": lam}

# Utilidades compartilhadas

def to_uniform(U: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Garante que U seja np.ndarray float64 em (0,1)"""
    if isinstance(U, pd.DataFrame):
        U = U.values
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    return np.clip(U, 1e-9, 1 - 1e-9)


def nearest_psd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Projeta matriz para o cone PSD mais próximo"""
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(A_psd))
    d = np.maximum(d, 1e-10)
    return A_psd / np.outer(d, d)


def spearman_to_pearson(rho_s: float) -> float:
    """Converte correlação de Spearman para correlação de Pearson da cópula Gaussiana"""
    return 2 * np.sin(np.pi * rho_s / 6)


def kendall_to_pearson(tau: float) -> float:
    """Converte tau de Kendall para correlação da cópula Gaussiana"""
    return np.sin(np.pi * tau / 2)


# Gaussiana

class GaussianCopula:
    """
    Cópula Gaussiana multivariada.

    Definição:
        C(u₁,...,uₙ) = Φₙ(Φ⁻¹(u₁),...,Φ⁻¹(uₙ); R)
    onde Φₙ é a CDF normal multivariada com matriz de correlação R.

    Dependência de cauda: λ_L = λ_U = 0 

    """

    def __init__(self, method: str = "spearman"):
        self.method = method
        self.R_: Optional[np.ndarray] = None      # Matriz de correlação
        self.n_dim_: int = 0
        self.n_obs_: int = 0
        self.log_likelihood_: float = np.nan
        self.aic_: float = np.nan
        self.bic_: float = np.nan
        self._is_fitted = False

    # Fit

    def fit(self, U: Union[np.ndarray, pd.DataFrame]) -> "GaussianCopula":
        """
        Estima a matriz de correlação da cópula Gaussiana
        """
        U = to_uniform(U)
        T, d = U.shape
        self.n_dim_ = d
        self.n_obs_ = T

        # Transformar para escala normal
        Z = stats.norm.ppf(U)

        if self.method == "mle":
            R = self._fit_mle(Z)
        elif self.method == "spearman":
            # Correlação de Spearman das uniformes → correlação Pearson da cópula
            rho_s = pd.DataFrame(U).corr(method="spearman").values
            R = np.vectorize(spearman_to_pearson)(rho_s)
            np.fill_diagonal(R, 1.0)
        elif self.method == "kendall":
            tau_matrix = np.zeros((d, d))
            for i in range(d):
                for j in range(i+1, d):
                    tau, _ = stats.kendalltau(U[:, i], U[:, j])
                    rho = kendall_to_pearson(tau)
                    tau_matrix[i, j] = rho
                    tau_matrix[j, i] = rho
            np.fill_diagonal(tau_matrix, 1.0)
            R = tau_matrix
        else:
            raise ValueError(f"método desconhecido: {self.method}")

        self.R_ = nearest_psd(R)
        self._compute_likelihood(Z)
        self._is_fitted = True

        logger.debug(
            f"GaussianCopula ajustada  d={d}  T={T}  "
            f"rho_mean={self._rho_mean():.3f}  LL={self.log_likelihood_:.2f}"
        )
        return self

    def _fit_mle(self, Z: np.ndarray) -> np.ndarray:
        """MLE via estimador amostral"""
        return np.corrcoef(Z.T)

    def _compute_likelihood(self, Z: np.ndarray):
        """Calcula log-likelihood, AIC e BIC"""
        T, d = Z.shape
        try:
            sign, log_det = np.linalg.slogdet(self.R_)
            if sign <= 0:
                self.log_likelihood_ = -np.inf
                return
            R_inv = np.linalg.inv(self.R_)
            # Log-density da cópula Gaussiana
            log_dens = (
                -0.5 * log_det
                - 0.5 * np.einsum("ti,ij,tj->t", Z, R_inv - np.eye(d), Z)
            )
            self.log_likelihood_ = float(np.sum(log_dens))
        except np.linalg.LinAlgError:
            self.log_likelihood_ = -np.inf

        d = self.n_dim_
        k = d * (d - 1) // 2  # parâmetros independentes da correlação
        self.aic_ = -2 * self.log_likelihood_ + 2 * k
        self.bic_ = -2 * self.log_likelihood_ + k * np.log(T)

    def _rho_mean(self) -> float:
        if self.R_ is None:
            return np.nan
        d = self.R_.shape[0]
        idx = np.triu_indices(d, k=1)
        return float(np.mean(self.R_[idx]))

    
    # Simulação

    def simulate(self, n_sim: int = 10000, seed: Optional[int] = None) -> np.ndarray:
        """
        Simula n_sim amostras da cópula Gaussiana
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        if seed is not None:
            np.random.seed(seed)

        L = np.linalg.cholesky(self.R_)
        Z = np.random.standard_normal((n_sim, self.n_dim_))
        Z_corr = Z @ L.T
        return stats.norm.cdf(Z_corr).astype(np.float32)

    # Log-density 

    def log_density(self, U: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Log-densidade da cópula Gaussiana em U
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        U = to_uniform(U)
        Z = stats.norm.ppf(U)

        sign, log_det = np.linalg.slogdet(self.R_)
        R_inv = np.linalg.inv(self.R_)
        d = self.n_dim_

        log_dens = (
            -0.5 * log_det
            - 0.5 * np.einsum("ti,ij,tj->t", Z, R_inv - np.eye(d), Z)
        )
        return log_dens

    # Bivariate conditional CDF
    def h_function(
        self, u: np.ndarray, v: np.ndarray, rho: Optional[float] = None
    ) -> np.ndarray:
        """
        h-function para par bivariado.
        h(uv) = P(U₁ ≤ u  U₂ = v

        Parametros:
        u, v : np.ndarray (T,), pseudo-observações do par
        rho  : float, opcional, correlação (usa self.R_[0,1] se bivariada)
        """
        if rho is None:
            if self.R_ is not None and self.n_dim_ == 2:
                rho = float(self.R_[0, 1])
            else:
                raise ValueError("Forneça rho para h_function bivariada.")

        u = np.clip(u, 1e-9, 1-1e-9)
        v = np.clip(v, 1e-9, 1-1e-9)

        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)

        num = z_u - rho * z_v
        den = np.sqrt(max(1 - rho**2, 1e-10))
        return stats.norm.cdf(num / den)

    # Tail dependence

    def tail_dependence(self) -> Dict[str, float]:
        """
        Coeficientes de dependência de cauda.
        Para cópula Gaussiana: λ_L = λ_U = 0
        """
        return {"lower_tail": 0.0, "upper_tail": 0.0}

    @property
    def rho_matrix(self) -> Optional[np.ndarray]:
        return self.R_

# t-student

class StudentTCopula:
    """
    Cópula t-Student multivariada.

    Definição:
        C(u₁,...,uₙ; R, ν) = tₙ(t_ν⁻¹(u₁),...,t_ν⁻¹(uₙ); R, ν)

    Parametros:
        'mle': estimação conjunta (R, ν) por MLE 
        'ifm': inference functions for margins
    nu_bounds : tupla
        Limites do grau de liberdade nu. 
    """
    _NU_IDENTIFICATION_WARNING = (
        "nu da copula t nao e identificavel via LL da copula (Genest 2012) Use AIC/BIC para escolher entre t e Gaussiana. Nu estimado pode ser maior que o verdadeiro em amostras finitas."
    )

    def __init__(
        self,
        method: str = "mle",
        nu_bounds: Tuple[float, float] = (2.0, 30.0),
    ):
        self.method = method
        self.nu_bounds = nu_bounds

        self.R_: Optional[np.ndarray] = None
        self.nu_: float = np.nan       # grau de liberdade
        self.n_dim_: int = 0
        self.n_obs_: int = 0
        self.log_likelihood_: float = np.nan
        self.aic_: float = np.nan
        self.bic_: float = np.nan
        self._is_fitted = False

   
    # Fit

    def fit(self, U: Union[np.ndarray, pd.DataFrame]) -> "StudentTCopula":
        """
        Estima (R, ν) da cópula t-Student.

        Parametros
        U : np.ndarray ou pd.DataFrame (T, d), pseudo-observações [0,1]
        """
        U = to_uniform(U)
        T, d = U.shape
        self.n_dim_ = d
        self.n_obs_ = T

        if self.method == "mle":
            R, nu = self._fit_mle(U)
        else:
            R, nu = self._fit_ifm(U)

        self.R_ = nearest_psd(R)
        self.nu_ = nu
        self._compute_likelihood(U)
        self._is_fitted = True

        td = self.tail_dependence()
        logger.debug(
            f"StudentTCopula ajustada  d={d}  ν={nu:.2f}  "
            f"LL={self.log_likelihood_:.2f} λ_L={td['lower_tail']:.4f}"
        )
        return self

    def _fit_mle(self, U: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        MLE conjunta de R e nu via ECME iterativo com multiplos starts
        Evita convergencia prematura no bound superior de nu
        """
        d = U.shape[1]

        # R inicial via Spearman
        rho_s  = pd.DataFrame(U).corr(method="spearman").values
        R_init = np.vectorize(spearman_to_pearson)(rho_s)
        np.fill_diagonal(R_init, 1.0)
        R_init = nearest_psd(R_init)

        # Estimar nu inicial via kurtosis e joint LL
        nu_kurtosis = self._estimate_nu_from_kurtosis(U)
        nu_joint    = self._estimate_nu_joint_ll(U, R_init) if U.shape[1] == 2 else nu_kurtosis
        # Grid centrado nas estimativas
        _nu_set = {nu_kurtosis, nu_joint,
                   max(self.nu_bounds[0], nu_kurtosis-2),
                   min(self.nu_bounds[1], nu_joint+3),
                   5.0, 10.0}
        nu_candidates = sorted(_nu_set)
        best_ll = -np.inf
        best_nu = 5.0
        best_R  = R_init.copy()

        for nu_init in nu_candidates:
            if nu_init < self.nu_bounds[0] or nu_init > self.nu_bounds[1]:
                continue
            try:
                R_cur  = R_init.copy()
                nu_cur = float(nu_init)

                for _ in range(10):  # ECME: max 10 iteracoes
                    t_sc = stats.t.ppf(np.clip(U, 1e-9, 1-1e-9), df=nu_cur)
                    try:
                        R_inv = np.linalg.inv(R_cur)
                    except np.linalg.LinAlgError:
                        break
                    quad = np.einsum("ti,ij,tj->t", t_sc, R_inv, t_sc)
                    W    = (nu_cur + d) / (nu_cur + quad)

                    R_new = np.cov(t_sc.T, aweights=W)
                    std   = np.maximum(np.sqrt(np.diag(R_new)), 1e-10)
                    R_new = R_new / np.outer(std, std)
                    R_cur = nearest_psd(R_new)

                    def _neg_nu(log_nu, _R=R_cur):
                        return -self._copula_log_likelihood(U, _R, np.exp(log_nu))

                    res = minimize_scalar(
                        _neg_nu,
                        bounds=(np.log(self.nu_bounds[0]), np.log(self.nu_bounds[1])),
                        method="bounded",
                        options={"xatol": 1e-4},
                    )
                    nu_cur = float(np.exp(res.x))

                ll_cur = self._copula_log_likelihood(U, R_cur, nu_cur)
                if ll_cur > best_ll:
                    best_ll = ll_cur
                    best_nu = nu_cur
                    best_R  = R_cur.copy()

            except Exception:
                continue

        if best_nu >= self.nu_bounds[1] * 0.80:
            logger.info(
                f"StudentTCopula: nu={best_nu:.1f}. "
                f"u da copula t nao e identificavel via LL da copula "
                f"(LL e monotonica em nu — Genest et al. 2012). "
                f"Use AIC para comparar t  Gaussiana em vez de interpretar nu diretamente."
            )
        elif best_nu <= self.nu_bounds[0] * 1.1:
            logger.warning(f"StudentTCopula: nu={best_nu:.1f} proximo do bound inferior.")

        return best_R, best_nu

    def _fit_ifm(self, U: np.ndarray) -> Tuple[np.ndarray, float]:
        """IFM mais rápido: Spearman para R, profile likelihood para ν."""
        d = U.shape[1]

        rho_s = pd.DataFrame(U).corr(method="spearman").values
        R = np.vectorize(spearman_to_pearson)(rho_s)
        np.fill_diagonal(R, 1.0)
        R = nearest_psd(R)

        def neg_ll_nu(nu):
            return -self._copula_log_likelihood(U, R, nu)

        res = minimize_scalar(
            neg_ll_nu,
            bounds=self.nu_bounds,
            method="bounded",
        )
        return R, float(res.x)

    def _copula_log_likelihood(
        self, U: np.ndarray, R: np.ndarray, nu: float
    ) -> float:
        """Log-likelihood da cópula t-Student."""
        T, d = U.shape
        try:
            t_scores = stats.t.ppf(np.clip(U, 1e-9, 1-1e-9), df=nu)

            sign, log_det = np.linalg.slogdet(R)
            if sign <= 0:
                return -1e10
            R_inv = np.linalg.inv(R)

            # Quadratic form: z'R⁻¹z
            quad = np.einsum("ti,ij,tj->t", t_scores, R_inv, t_scores)

            # Log-densidade da t multivariada
            ll_mv = (
                gammaln((nu + d) / 2)
                - gammaln(nu / 2)
                - (d / 2) * np.log(nu * np.pi)
                - 0.5 * log_det
                - ((nu + d) / 2) * np.log(1 + quad / nu)
            )

            # Subtrair log-densidades marginais t_ν para obter cópula
            log_marg = np.sum(
                stats.t.logpdf(t_scores, df=nu), axis=1
            )

            log_cop = ll_mv - log_marg
            return float(np.sum(log_cop))

        except Exception:
            return -1e10

    def _compute_likelihood(self, U: np.ndarray):
        T = U.shape[0]
        self.log_likelihood_ = self._copula_log_likelihood(U, self.R_, self.nu_)
        k = self.n_dim_ * (self.n_dim_ - 1) // 2 + 1  # correlações + nu
        self.aic_ = -2 * self.log_likelihood_ + 2 * k
        self.bic_ = -2 * self.log_likelihood_ + k * np.log(T)

    # Simulação

    def simulate(self, n_sim: int = 10000, seed: Optional[int] = None) -> np.ndarray:
        """
        Simula n_sim amostras da cópula t-Student

        Algoritmo: X = Z√(ν/χ²_ν), Z ~ N(0,R), χ²_ν ~ chi2(ν)

        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        if seed is not None:
            np.random.seed(seed)

        d = self.n_dim_
        nu = self.nu_

        L = np.linalg.cholesky(self.R_)
        Z = np.random.standard_normal((n_sim, d)) @ L.T

        chi2 = np.random.chisquare(nu, n_sim)
        T_samples = Z / np.sqrt(chi2[:, None] / nu)

        return stats.t.cdf(T_samples, df=nu).astype(np.float32)

    # Log-density e h-function
    
    def log_density(self, U: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Log-densidade da cópula t-Student em U."""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        U = to_uniform(U)
        T = U.shape[0]
        ll_total = np.zeros(T)

        t_scores = stats.t.ppf(U, df=self.nu_)
        sign, log_det = np.linalg.slogdet(self.R_)
        R_inv = np.linalg.inv(self.R_)
        d = self.n_dim_
        nu = self.nu_

        quad = np.einsum("ti,ij,tj->t", t_scores, R_inv, t_scores)
        ll_mv = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - (d / 2) * np.log(nu * np.pi)
            - 0.5 * log_det
            - ((nu + d) / 2) * np.log(1 + quad / nu)
        )
        log_marg = np.sum(stats.t.logpdf(t_scores, df=nu), axis=1)
        return ll_mv - log_marg

    def h_function(
        self, u: np.ndarray, v: np.ndarray,
        rho: Optional[float] = None,
        nu: Optional[float] = None,
    ) -> np.ndarray:
        """
        h-function bivariada da cópula t-Student
        h(uv) = t_{ν+1}((t_ν⁻¹(u) - ρ·t_ν⁻¹(v)) / √((ν + t_ν⁻¹(v)²)(1-ρ²)/(ν+1)))
        """
        if rho is None:
            rho = float(self.R_[0, 1]) if self.n_dim_ == 2 else 0.0
        if nu is None:
            nu = self.nu_

        u = np.clip(u, 1e-9, 1-1e-9)
        v = np.clip(v, 1e-9, 1-1e-9)

        t_u = stats.t.ppf(u, df=nu)
        t_v = stats.t.ppf(v, df=nu)

        num = t_u - rho * t_v
        den = np.sqrt((nu + t_v**2) * (1 - rho**2) / (nu + 1))
        den = np.maximum(den, 1e-10)
        return stats.t.cdf(num / den, df=nu + 1)

    # Tail dependence

    def tail_dependence(
        self,
        rho: Optional[float] = None,
        U: Optional[np.ndarray] = None,
        alpha: float = 0.10,
    ) -> Dict[str, float]:
        """
        Coeficientes de dependencia de cauda da copula t-Student

        Se U (pseudo-obs) fornecido usa estimador empirico cfg, nao depende de nu, que nao e identificavel via LL da copula
        Fallback analitico: 2*t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))) subestima tail dep quando nu esta superestimado (nu grande)

        Parametros
        rho: float, opcional, correlacao, usa media da matriz se None
        U: np.ndarray (T, d), opcional, pseudo-obs para estimativa empirica
        alpha: float, nivel de cauda para estimador empirico
        """
        if self.nu_ is None or self.R_ is None:
            return {"lower_tail": 0.0, "upper_tail": 0.0}

        # estimador empirico (nao depende de nu)
        if U is not None:
            emp = empirical_tail_dependence(U, alpha=alpha)
            logger.debug(
                f"StudentTCopula.tail_dependence: empirico "
                f"lambda_L={emp['lower_tail']:.4f} "
                f"lambda_U={emp['upper_tail']:.4f} (alpha={alpha})"
            )
            return emp

        # nalitico via nu (fallback)
        nu = self.nu_
        if rho is None:
            d = self.R_.shape[0]
            idx = np.triu_indices(d, k=1)
            rho = float(np.mean(self.R_[idx]))

        result = analytical_tail_dependence_t(nu, rho)
        logger.debug(
            f"StudentTCopula.tail_dependence: analitico nu={nu:.1f} "
            f"lambda={result['lower_tail']:.4f} "
            f"(nu pode estar superestimado, prefira passar U)"
        )
        return result

    @property
    def rho_matrix(self) -> Optional[np.ndarray]:
        return self.R_

# Cópula Gaussiana Bivariada

class BivariateGaussianCopula:
    """
    Versão bivariada
    Usada como pair-copula em vine_copulas.py, mais rápida que a multivariada.

    Parameter: rho ∈ (-1, 1)
    """

    def __init__(self):
        self.rho_: float = 0.0
        self.log_likelihood_: float = np.nan
        self.aic_: float = np.nan
        self.bic_: float = np.nan
        self._is_fitted = False

    def fit(self, u: np.ndarray, v: np.ndarray) -> "BivariateGaussianCopula":
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)

        # Estimativa inicial via Kendall
        tau, _ = stats.kendalltau(u, v)
        rho0 = np.clip(kendall_to_pearson(tau), -0.99, 0.99)

        def neg_ll(rho):
            if abs(rho) >= 1:
                return 1e10
            z_u = stats.norm.ppf(u)
            z_v = stats.norm.ppf(v)
            r2 = 1 - rho**2
            ll = (
                -0.5 * np.log(r2)
                - 0.5 / r2 * (z_u**2 + z_v**2 - 2*rho*z_u*z_v)
                + 0.5 * (z_u**2 + z_v**2)
            )
            return -np.sum(ll)

        res = minimize_scalar(
            neg_ll, bounds=(-0.999, 0.999), method="bounded"
        )
        self.rho_ = float(res.x)

        T = len(u)
        ll = -res.fun
        self.log_likelihood_ = ll
        self.aic_ = -2*ll + 2
        self.bic_ = -2*ll + np.log(T)
        self._is_fitted = True
        return self

    def simulate(self, n_sim: int) -> np.ndarray:
        z1 = np.random.standard_normal(n_sim)
        z2 = self.rho_ * z1 + np.sqrt(1 - self.rho_**2) * np.random.standard_normal(n_sim)
        return np.column_stack([stats.norm.cdf(z1), stats.norm.cdf(z2)]).astype(np.float32)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        return stats.norm.cdf((z_u - self.rho_ * z_v) / np.sqrt(max(1 - self.rho_**2, 1e-10)))

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)
        z_u = stats.norm.ppf(u)
        z_v = stats.norm.ppf(v)
        r2 = 1 - self.rho_**2
        return (-0.5*np.log(r2)
                - 0.5/r2*(z_u**2 + z_v**2 - 2*self.rho_*z_u*z_v)
                + 0.5*(z_u**2 + z_v**2))

    def tail_dependence(self) -> Dict[str, float]:
        return {"lower_tail": 0.0, "upper_tail": 0.0}

# 5. Cópula t-Student bivariada

class BivariateStudentTCopula:
    """
    Versão bivariada otimizada da cópula t-Student.
    Parâmetros: rho ∈ (-1,1), nu > 2
    """

    def __init__(self, nu_bounds: Tuple[float, float] = (2.0, 30.0)):
        self.rho_: float = 0.0
        self.nu_: float = 5.0
        self.nu_bounds = nu_bounds
        self.log_likelihood_: float = np.nan
        self.aic_: float = np.nan
        self.bic_: float = np.nan
        self._is_fitted = False

    def _neg_ll(self, params: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        rho, log_nu = params
        nu = np.exp(log_nu)
        if abs(rho) >= 1 or nu < self.nu_bounds[0] or nu > self.nu_bounds[1]:
            return 1e10

        t_u = stats.t.ppf(u, df=nu)
        t_v = stats.t.ppf(v, df=nu)
        r2 = 1 - rho**2

        ll = (
            gammaln((nu + 2) / 2)
            - gammaln(nu / 2)
            - np.log(nu * np.pi * r2) / 2
            - ((nu + 2) / 2) * np.log(1 + (t_u**2 + t_v**2 - 2*rho*t_u*t_v) / (nu * r2))
            - stats.t.logpdf(t_u, df=nu)
            - stats.t.logpdf(t_v, df=nu)
        )
        return -np.sum(ll)

    def _nu_from_kurtosis(self, u: np.ndarray, v: np.ndarray) -> float:
        """Nu via kurtosis: contorna identificabilidade da LL da copula t."""
        from scipy.stats import kurtosis as _kurt
        kurts = []
        for x in [u, v]:
            z = stats.norm.ppf(x)
            k = float(_kurt(z, fisher=True))
            if k > 0.2:
                kurts.append(np.clip(6.0/k + 4.0, self.nu_bounds[0], self.nu_bounds[1]))
        if kurts:
            return float(np.median(kurts))
        # Fallback: joint LL
        tau, _ = stats.kendalltau(u, v)
        rho_e = np.clip(kendall_to_pearson(tau), -0.99, 0.99)
        XY = np.column_stack([stats.norm.ppf(u), stats.norm.ppf(v)])
        def _nj(lnu):
            nu=np.exp(lnu); r2=1-rho_e**2
            q=(XY[:,0]**2-2*rho_e*XY[:,0]*XY[:,1]+XY[:,1]**2)/r2
            return -np.sum(gammaln((nu+2)/2)-gammaln(nu/2)-np.log(nu*np.pi*r2)-((nu+2)/2)*np.log(1+q/nu))
        res=minimize_scalar(_nj,bounds=(np.log(self.nu_bounds[0]),np.log(self.nu_bounds[1])),method='bounded')
        return float(np.exp(res.x))

    def fit(self, u: np.ndarray, v: np.ndarray) -> "BivariateStudentTCopula":
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)

        tau, _ = stats.kendalltau(u, v)
        rho0 = np.clip(kendall_to_pearson(tau), -0.99, 0.99)

        # Estimar nu via kurtosis para evitar bound trapping
        nu_init = self._nu_from_kurtosis(u, v)
        log_nu_init = np.log(np.clip(nu_init, self.nu_bounds[0]+0.01, self.nu_bounds[1]-0.01))

        # Multiplos starts em rho e nu
        best_ll, best_rho, best_nu = -np.inf, rho0, nu_init
        for lns in [log_nu_init, np.log(5.0), np.log(10.0), np.log(3.0)]:
            res = minimize(
                self._neg_ll,
                x0=[rho0, lns],
                args=(u, v),
                method="L-BFGS-B",
                bounds=[(-0.999, 0.999), (np.log(self.nu_bounds[0]), np.log(self.nu_bounds[1]))],
                options={"maxiter": 500},
            )
            if -res.fun > best_ll:
                best_ll = -res.fun; best_rho = float(res.x[0]); best_nu = float(np.exp(res.x[1]))
        # Se nu saturou, usar estimativa via kurtosis
        if best_nu >= self.nu_bounds[1] * 0.95:
            best_nu = nu_init
            logger.debug(f'BivT: nu saturou bound, forcando nu_kurtosis={nu_init:.1f}')
        res = type('R', (), {'x': [best_rho, np.log(best_nu)], 'fun': -best_ll})()
        self.rho_ = best_rho
        self.nu_  = best_nu

        T = len(u)
        ll = best_ll
        self.log_likelihood_ = ll
        self.aic_ = -2*ll + 2*2
        self.bic_ = -2*ll + 2*np.log(T)
        self._is_fitted = True
        return self

    def simulate(self, n_sim: int) -> np.ndarray:
        nu = self.nu_
        rho = self.rho_
        z1 = np.random.standard_normal(n_sim)
        z2 = rho*z1 + np.sqrt(1-rho**2)*np.random.standard_normal(n_sim)
        chi2 = np.random.chisquare(nu, n_sim)
        t1 = z1 / np.sqrt(chi2/nu)
        t2 = z2 / np.sqrt(chi2/nu)
        return np.column_stack([
            stats.t.cdf(t1, df=nu),
            stats.t.cdf(t2, df=nu)
        ]).astype(np.float32)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h-function bivariada t-Student para vine_copulas.py"""
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)
        nu, rho = self.nu_, self.rho_
        t_u = stats.t.ppf(u, df=nu)
        t_v = stats.t.ppf(v, df=nu)
        num = t_u - rho * t_v
        den = np.sqrt((nu + t_v**2) * (1 - rho**2) / (nu + 1))
        return stats.t.cdf(num / np.maximum(den, 1e-10), df=nu+1)

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Log-densidade da copula t-Student bivariada"""
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)
        nu, rho = self.nu_, self.rho_
        t_u = stats.t.ppf(u, df=nu)
        t_v = stats.t.ppf(v, df=nu)
        r2  = 1 - rho**2
        return (
            gammaln((nu+2)/2) - gammaln(nu/2)
            - 0.5*np.log(nu*np.pi*r2)
            - ((nu+2)/2)*np.log(1 + (t_u**2 + t_v**2 - 2*rho*t_u*t_v) / (nu*r2))
            - stats.t.logpdf(t_u, df=nu)
            - stats.t.logpdf(t_v, df=nu)
        )

    def tail_dependence(
        self,
        u: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        alpha: float = 0.10,
    ) -> Dict[str, float]:
        """
        Tail dependence da copula t bivariada

        Prioridade:
        Se (u, v) fornecidos: estimador empirico cfg nao depende de nu 
        Fallback analitico via nu estimado

        Parametros
        u, v: np.ndarray (T,), opcional, seudo-obs do par
        alpha: float, nivel de cauda para estimador empirico
        """
        # Prioridade 1: empirico
        if u is not None and v is not None:
            U_pair = np.column_stack([
                np.clip(u.ravel(), 1e-9, 1-1e-9),
                np.clip(v.ravel(), 1e-9, 1-1e-9),
            ])
            emp = empirical_tail_dependence(U_pair, alpha=alpha)
            logger.debug(
                f"BivariateT.tail_dependence: empirico "
                f"lL={emp['lower_tail']:.4f} lU={emp['upper_tail']:.4f}"
            )
            return emp

        # Prioridade 2: analitico (fallback)
        result = analytical_tail_dependence_t(self.nu_, self.rho_)
        logger.debug(
            f"BivariateT.tail_dependence: analitico nu={self.nu_:.1f} "
            f"lambda={result['lower_tail']:.4f} "
            f"(prefira passar u,v para estimativa empirica)"
        )
        return result


# MAIN

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    # T maior e nu menor tornam as caudas pesadas identificaveis
    T = 1500
    d = 4
    true_rho = 0.6
    true_nu  = 4.0   # nu=4: caudas claramente pesadas (kurtosis=6)
    L = np.linalg.cholesky([[1, true_rho], [true_rho, 1]])
    Z = np.random.standard_normal((T, 2)) @ L.T
    chi2 = np.random.chisquare(true_nu, T)
    T_samp = Z / np.sqrt(chi2[:,None] / true_nu)
    U_biv = stats.t.cdf(T_samp, df=true_nu)

    # Gaussiana bivariada
    print("\n BivariateGaussianCopula ")
    gauss = BivariateGaussianCopula()
    gauss.fit(U_biv[:,0], U_biv[:,1])
    print(f"rho estimado: {gauss.rho_:.4f} (verdadeiro: {true_rho})")
    print(f"AIC={gauss.aic_:.2f}  BIC={gauss.bic_:.2f}")

    # t-Student bivariada
    print("\n BivariateStudentTCopula ")
    student = BivariateStudentTCopula()
    student.fit(U_biv[:,0], U_biv[:,1])
    print(f"rho={student.rho_:.4f} (verdadeiro: {true_rho})")
    print(f"nu={student.nu_:.2f} (verdadeiro: {true_nu})")
    print(f"AIC={student.aic_:.2f}  BIC={student.bic_:.2f}")
    # Tail dep analitica (via nu estimado — pode subestimar)
    print(f"Tail dep analitica (nu={student.nu_:.1f}): {student.tail_dependence()}")
    # Tail dep empirica (nao depende de nu — fonte primaria)
    print(f"Tail dep empirica  (CFG alpha=0.10): {student.tail_dependence(u=U_biv[:,0], v=U_biv[:,1])}")
    # Comparar AIC: t  Gaussiana
    print(f"AIC t={student.aic_:.2f}  AIC Gaussiana={gauss.aic_:.2f} -> {'t-Student' if student.aic_ < gauss.aic_ else 'Gaussiana'}")

    # Multivariada
    print("\n GaussianCopula multivariada (d=4) ")
    # Simular 4 ativos correlacionados
    R_true = 0.3*np.ones((d,d)) + 0.7*np.eye(d)
    L4 = np.linalg.cholesky(R_true)
    Z4 = np.random.standard_normal((T,d)) @ L4.T
    U4 = stats.norm.cdf(Z4)

    gc = GaussianCopula(method="spearman")
    gc.fit(U4)
    print(f"rho_mean={gc._rho_mean():.4f}  LL={gc.log_likelihood_:.2f}")
    U_sim = gc.simulate(1000)
    print(f"Simulação: {U_sim.shape}  [0,1]: {U_sim.min():.3f} - {U_sim.max():.3f}")

    print("\n StudentTCopula multivariada (d=4) ")
    tc = StudentTCopula(method="ifm")
    tc.fit(U4)
    print(f"nu={tc.nu_:.2f}  LL={tc.log_likelihood_:.2f}")
    # Tail dep analitica  empirica
    print(f"Tail dep analitica (nu={tc.nu_:.1f}): {tc.tail_dependence()}")
    print(f"Tail dep empirica  (CFG alpha=0.10): {tc.tail_dependence(U=U4)}")
    # Selecao por AIC
    print(f"AIC t={tc.aic_:.2f} AIC Gaussiana={gc.aic_:.2f} -> {'t-Student' if tc.aic_ < gc.aic_ else 'Gaussiana'}")

    # Funcao standalone
    print("\n empirical_tail_dependence() standalone ")
    td_emp = empirical_tail_dependence(U4, alpha=0.10)
    print(f"lambda_L={td_emp['lower_tail']:.4f}  lambda_U={td_emp['upper_tail']:.4f}  pares={td_emp['n_pairs']}")

    print("\n Teste concluido.")
