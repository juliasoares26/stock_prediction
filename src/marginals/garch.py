import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch não disponível. Usando implementação própria de GARCH(1,1)")

@dataclass
class GARCHResult:
    """Resultado do ajuste GARCH para um único ativo"""
    ticker: str
    model_type: str                          # 'GARCH', 'EGARCH', 'GJR'
    params: Dict[str, float]                 # omega, alpha, beta, [gamma, ...]
    conditional_vol: pd.Series               # sigma_t (T,)
    std_residuals: pd.Series                 # z_t = r_t / sigma_t (T,)
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_obs: int

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"Modelo: {self.model_type}  Ticker: {self.ticker}",
            f"{'='*50}",
            f"Parâmetros:",
        ]
        for k, v in self.params.items():
            lines.append(f"  {k:12s} = {v:.6f}")
        lines += [
            f"Log-likelihood : {self.log_likelihood:.4f}",
            f"AIC            : {self.aic:.4f}",
            f"BIC            : {self.bic:.4f}",
            f"Convergiu      : {self.converged}",
            f"Observações    : {self.n_obs}",
        ]
        return "\n".join(lines)


class _GARCH11:
    """
    GARCH(1,1) implementado via MLE com scipy
    Modelo: r_t = sigma_t * z_t,  z_t ~ N(0,1)
        sigma²_t = omega + alpha * r²_{t-1} + beta * sigma²_{t-1}
    """

    def __init__(self):
        self.params_ = None
        self.sigma2_ = None

    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        omega, alpha, beta = params
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        return np.maximum(sigma2, 1e-10)

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        sigma2 = self._compute_variance(params, returns)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        return -ll

    def fit(self, returns: np.ndarray) -> "_GARCH11":
        var = np.var(returns)
        x0 = np.array([var * 0.05, 0.08, 0.88])
        bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(returns,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        self.params_ = result.x
        self.sigma2_ = self._compute_variance(self.params_, returns)
        self._converged = result.success
        self._ll = -result.fun
        return self

    def get_result(
        self, ticker: str, index: pd.Index, returns: np.ndarray
    ) -> GARCHResult:
        omega, alpha, beta = self.params_
        sigma = np.sqrt(self.sigma2_)
        z = returns / sigma

        T = len(returns)
        k = 3
        ll = self._ll
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(T)

        return GARCHResult(
            ticker=ticker,
            model_type="GARCH(1,1)",
            params={"omega": omega, "alpha": alpha, "beta": beta,
                    "persistence": alpha + beta},
            conditional_vol=pd.Series(sigma, index=index, name=f"{ticker}_vol"),
            std_residuals=pd.Series(z, index=index, name=f"{ticker}_z"),
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            converged=self._converged,
            n_obs=T,
        )


class _GJRGARCH11:
    """
    GJR-GARCH(1,1) com efeito assimétrico (alavancagem)
    sigma²_t = omega + (alpha + gamma * I_{r<0}) * r²_{t-1} + beta * sigma²_{t-1}
    """

    def __init__(self):
        self.params_ = None
        self.sigma2_ = None

    def _compute_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        omega, alpha, gamma, beta = params
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)

        for t in range(1, T):
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            sigma2[t] = (omega
                         + (alpha + gamma * indicator) * returns[t-1]**2
                         + beta * sigma2[t-1])

        return np.maximum(sigma2, 1e-10)

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
            return 1e10
        if alpha + 0.5 * gamma + beta >= 1:
            return 1e10

        sigma2 = self._compute_variance(params, returns)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)
        return -ll

    def fit(self, returns: np.ndarray) -> "_GJRGARCH11":
        var = np.var(returns)
        x0 = np.array([var * 0.05, 0.05, 0.08, 0.85])
        bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.999)]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(returns,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        self.params_ = result.x
        self.sigma2_ = self._compute_variance(self.params_, returns)
        self._converged = result.success
        self._ll = -result.fun
        return self

    def get_result(
        self, ticker: str, index: pd.Index, returns: np.ndarray
    ) -> GARCHResult:
        omega, alpha, gamma, beta = self.params_
        sigma = np.sqrt(self.sigma2_)
        z = returns / sigma

        T = len(returns)
        k = 4
        ll = self._ll
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(T)

        return GARCHResult(
            ticker=ticker,
            model_type="GJR-GARCH(1,1)",
            params={"omega": omega, "alpha": alpha, "gamma": gamma,
                    "beta": beta, "persistence": alpha + 0.5*gamma + beta},
            conditional_vol=pd.Series(sigma, index=index, name=f"{ticker}_vol"),
            std_residuals=pd.Series(z, index=index, name=f"{ticker}_z"),
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            converged=self._converged,
            n_obs=T,
        )

class GARCHFitter:
    """
    Ajusta modelos GARCH para todos os ativos de um portfólio.

    Usa a biblioteca 'arch' se disponível com fallback para implementação própria GARCH(1,1) e GJR-GARCH(1,1)

    """

    def __init__(
        self,
        model_type: str = "gjr",
        dist: str = "normal",
        p: int = 1,
        q: int = 1,
    ):
        self.model_type = model_type.lower()
        self.dist = dist
        self.p = p
        self.q = q

        self.results_: Dict[str, GARCHResult] = {}
        self._is_fitted = False

    # Fit único ativo

    def fit_single(
        self,
        returns: pd.Series,
        ticker: Optional[str] = None,
    ) -> GARCHResult:
        """
        Ajusta GARCH para uma única série de retornos
        """
        ticker = ticker or str(returns.name or "ativo")
        r = returns.dropna().values * 100  # escalar para % melhora convergência

        if ARCH_AVAILABLE:
            result = self._fit_arch(r, ticker, returns.dropna().index)
        else:
            result = self._fit_own(r, ticker, returns.dropna().index)

        # Desescalar volatilidade
        result.conditional_vol = result.conditional_vol / 100
        result.std_residuals = result.std_residuals  # z_t é adimensional

        # alpha pode ter chave 'alpha' (proprio/GJR) ou 'alpha[1]' (arch lib)
        _alpha = result.params.get('alpha', result.params.get('alpha[1]', np.nan))
        _beta  = result.params.get('beta',  result.params.get('beta[1]',  np.nan))
        _gamma = result.params.get('gamma', None)
        _g_str = f" gamma={_gamma:.4f}" if _gamma is not None else ""
        logger.info(
            f"{ticker:15s}  {result.model_type}  "
            f"alpha={_alpha:.4f}{_g_str} "
            f"beta={_beta:.4f} "
            f"persist={result.params.get('persistence',np.nan):.4f}  "
            f"AIC={result.aic:.2f}"
        )
        return result

    def _fit_arch(
        self, r: np.ndarray, ticker: str, index: pd.Index
    ) -> GARCHResult:
        """Usa biblioteca arch."""
        mtype = self.model_type
        if mtype == "auto":
            mtype = "gjr"  # resolve no fit_all

        vol_model = {"garch": "GARCH", "gjr": "GARCH", "egarch": "EGARCH"}.get(mtype, "GARCH")
        power = 2.0
        o = 1 if mtype == "gjr" else 0

        try:
            am = arch_model(
                r, vol=vol_model, p=self.p, o=o, q=self.q,
                dist=self.dist, power=power, rescale=False
            )
            fit = am.fit(disp="off", show_warning=False, options={"maxiter": 500})

            sigma = fit.conditional_volatility / 100  # desescalar
            z = (r / 100) / sigma

            params_dict = dict(fit.params)
            # Normalizar nomes: arch usa 'alpha[1]'/'beta[1]', nos usamos 'alpha'/'beta'
            if 'alpha[1]' in params_dict:
                params_dict['alpha'] = params_dict.pop('alpha[1]')
            if 'beta[1]' in params_dict:
                params_dict['beta'] = params_dict.pop('beta[1]')
            if 'gamma[1]' in params_dict:
                params_dict['gamma'] = params_dict.pop('gamma[1]')
            if 'omega' not in params_dict and 'omega' in str(list(params_dict.keys())):
                pass  # omega ja tem nome correto
            params_dict['persistence'] = (
                params_dict.get('alpha', 0)
                + params_dict.get('beta', 0)
                + 0.5 * params_dict.get('gamma', 0)
            )

            T = len(r)
            k = len(fit.params)
            ll = fit.loglikelihood

            return GARCHResult(
                ticker=ticker,
                model_type=f"{vol_model}({self.p},{self.q})",
                params=params_dict,
                conditional_vol=pd.Series(sigma, index=index, name=f"{ticker}_vol"),
                std_residuals=pd.Series(z, index=index, name=f"{ticker}_z"),
                log_likelihood=ll,
                aic=-2 * ll + 2 * k,
                bic=-2 * ll + k * np.log(T),
                converged=True,
                n_obs=T,
            )
        except Exception as e:
            logger.warning(f"{ticker}: arch falhou ({e}). Usando fallback.")
            return self._fit_own(r, ticker, index)

    def _fit_own(
        self, r: np.ndarray, ticker: str, index: pd.Index
    ) -> GARCHResult:
        """Implementação própria."""
        mtype = self.model_type
        if mtype in ("gjr", "auto", "egarch"):
            g = _GJRGARCH11()
        else:
            g = _GARCH11()

        g.fit(r / 100)  # normalizar para retornos decimais
        result = g.get_result(ticker, index, r / 100)
        return result

    # Fit todos os ativos

    def fit_all(
        self,
        returns_df: pd.DataFrame,
        select_best: bool = False,
    ) -> Dict[str, GARCHResult]:
        """
        Ajusta GARCH para todos os ativos do DataFrame
        """
        
        logger.info(f"Ajustando GARCH para {len(returns_df.columns)} ativos")

        for ticker in returns_df.columns:
            series = returns_df[ticker].dropna()
            if len(series) < 50:
                logger.warning(f"{ticker}: dados insuficientes ({len(series)}). Pulando.")
                continue

            if self.model_type == "auto" and select_best:
                result = self._auto_select(series, ticker)
            else:
                result = self.fit_single(series, ticker)

            self.results_[ticker] = result

        self._is_fitted = True
        logger.info(f"GARCH ajustado para {len(self.results_)} ativos")
        return self.results_

    def _auto_select(self, series: pd.Series, ticker: str) -> GARCHResult:
        """Seleciona GARCH ou GJR pelo menor AIC."""
        r_garch = GARCHFitter(model_type="garch", dist=self.dist).fit_single(series, ticker)
        r_gjr   = GARCHFitter(model_type="gjr",   dist=self.dist).fit_single(series, ticker)
        best = r_garch if r_garch.aic < r_gjr.aic else r_gjr
        logger.info(f"{ticker}: selecionado {best.model_type} (AIC={best.aic:.2f})")
        return best

    # Outputs para integração

    def get_std_residuals(self) -> pd.DataFrame:
        """
        Retorna DataFrame (T, n_ativos) de resíduos padronizados z_t
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        return pd.DataFrame({t: r.std_residuals for t, r in self.results_.items()})

    def get_conditional_vol(self) -> pd.DataFrame:
        """
        Retorna DataFrame (T, n_ativos) de volatilidades condicionais sigma_t
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        return pd.DataFrame({t: r.conditional_vol for t, r in self.results_.items()})

    def get_params_df(self) -> pd.DataFrame:
        """
        Retorna DataFrame com parâmetros estimados por ativo
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        rows = []
        for ticker, r in self.results_.items():
            row = {"ticker": ticker, "model": r.model_type,
                   "aic": r.aic, "bic": r.bic, "ll": r.log_likelihood,
                   "converged": r.converged}
            row.update(r.params)
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")

    def get_persistence(self) -> pd.Series:
        """Retorna persistência (alpha+beta) por ativo"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        return pd.Series(
            {t: r.params.get("persistence", np.nan) for t, r in self.results_.items()},
            name="persistence"
        )


def fit_garch_all(
    returns_df: pd.DataFrame,
    model_type: str = "gjr",
    dist: str = "normal",
    save_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Wrapper para main_pipeline.py
    """
    fitter = GARCHFitter(model_type=model_type, dist=dist)
    results = fitter.fit_all(returns_df)

    std_resid   = fitter.get_std_residuals()
    cond_vol    = fitter.get_conditional_vol()

    if save_path:
        from pathlib import Path
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        std_resid.to_parquet(p / "garch_std_residuals.parquet")
        cond_vol.to_parquet(p / "garch_conditional_vol.parquet")
        fitter.get_params_df().to_csv(p / "garch_params.csv")
        logger.info(f"GARCH outputs salvos em {save_path}")

    return std_resid, cond_vol, results

# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 60)
    print("TESTE: GARCHFitter com retornos sintéticos")
    print("=" * 60)

    np.random.seed(42)
    T = 800
    n = 5
    dates = pd.date_range("2020-01-01", periods=T, freq="B")

    # Simular GJR-GARCH(1,1)
    # vol diaria ~1.5%  omega calibrado para variancia incondicional ~(0.015)^2
    omega, alpha, gamma, beta = 3e-6, 0.05, 0.08, 0.88
    # persistencia = alpha + 0.5*gamma + beta = 0.97
    returns_sim = np.zeros((T, n))
    sigma2 = np.full(n, omega / (1 - alpha - 0.5*gamma - beta))

    for t in range(T):
        sigma = np.sqrt(sigma2)
        r = sigma * np.random.standard_normal(n)
        returns_sim[t] = r
        indicator = (r < 0).astype(float)
        sigma2 = omega + (alpha + gamma*indicator) * r**2 + beta * sigma2

    returns_df = pd.DataFrame(
        returns_sim,
        index=dates,
        columns=[f"Ativo{i+1}" for i in range(n)]
    )

    print(f"returns_df: {returns_df.shape}")

    # Fit
    std_resid, cond_vol, garch_results = fit_garch_all(
        returns_df, model_type="gjr"
    )

    print(f"\nResíduos padronizados: {std_resid.shape}")
    print(f"Vol condicional: {cond_vol.shape}")

    # Verificar normalidade dos resíduos
    print("\nTeste de normalidade dos resíduos (Jarque-Bera):")
    for col in std_resid.columns:
        z = std_resid[col].dropna()
        jb, pval = stats.jarque_bera(z)
        print(f"  {col}: JB={jb:.2f}, p={pval:.4f} {'normal' if pval > 0.05 else 'não-normal'}")

    # Persistência
    fitter = GARCHFitter(model_type="gjr")
    fitter.fit_all(returns_df)
    print(f"\nPersistência GARCH:\n{fitter.get_persistence().round(4)}")
    print(f"\nParâmetros:\n{fitter.get_params_df()[['model','aic','persistence']].round(4)}")

    print("\n Teste concluído.")
