import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gamma as gamma_func

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GEVResult:
    """Resultado do ajuste GEV para um único ativo."""
    ticker: str
    xi: float      # shape — <0: Weibull (cauda finita), =0: Gumbel, >0: Fréchet
    mu: float      # location
    sigma: float   # scale (> 0)
    n_blocks: int
    block_size: int
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    block_maxima: np.ndarray   # série de máximos extraídos

    @property
    def gev_type(self) -> str:
        if abs(self.xi) < 0.05:
            return "Gumbel (xi≈0)"
        elif self.xi > 0:
            return f"Fréchet (xi={self.xi:.3f}) — cauda pesada"
        else:
            return f"Weibull (xi={self.xi:.3f}) — cauda finita"

    def return_level(self, return_period: float) -> float:
        """
        Nível de retorno para um período dado

        """
        p = 1 - 1 / return_period
        if abs(self.xi) < 1e-6:
            # Gumbel
            return self.mu - self.sigma * np.log(-np.log(p))
        else:
            return self.mu + (self.sigma / self.xi) * ((-np.log(p))**(-self.xi) - 1)

    def summary(self) -> str:
        lines = [
            f"GEV  Ticker: {self.ticker}",
            f"Tipo: {self.gev_type}",
            f"xi (shape): {self.xi:.6f}",
            f"mu (location): {self.mu:.6f}",
            f"sigma (scale): {self.sigma:.6f}",
            f"Blocos: {self.n_blocks} (tamanho={self.block_size})",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            f"Convergiu: {self.converged}",
        ]
        return "\n".join(lines)

# Extração de Block Maxima

def extract_block_maxima(
    series: pd.Series,
    block_size: int = 22,
    use_minima: bool = True,
) -> pd.Series:
    """
    Extrai máximos ou minimos de blocos não sobrepostos.
    """
    s = series.dropna()
    T = len(s)
    n_blocks = T // block_size

    if n_blocks < 10:
        logger.warning(
            f"Apenas {n_blocks} blocos com block_size={block_size}. "
            f"Considere reduzir block_size ou usar mais dados."
        )

    block_extremes = []
    block_dates = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block = s.iloc[start:end]

        if use_minima:
            # Mínimo do bloco = pior retorno = máximo da perda
            extreme = block.min()
        else:
            extreme = block.max()

        block_extremes.append(float(extreme))
        block_dates.append(block.index[-1])

    result = pd.Series(block_extremes, index=block_dates, name=f"block_extreme_{series.name}")

    logger.debug(
        f"Block maxima: {n_blocks} blocos  "
        f"média={result.mean():.4f}  min={result.min():.4f}"
    )
    return result

# AJUSTE GEV (MLE)

class GEVFitter:
    """
    Ajusta a distribuição GEV via Maximum Likelihood Estimation.
    xi > 0: Fréchet  (caudas pesadas)
    xi = 0: Gumbel   (caudas exponenciais)
    xi < 0: Weibull  (cauda finita)

    """

    def __init__(self):
        self.results_: Dict[str, GEVResult] = {}
        self._is_fitted = False

  
    # Log-likelihood 

    @staticmethod
    def _gev_log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
        xi, mu, sigma = params
        if sigma <= 0:
            return 1e10

        n = len(data)
        z = (data - mu) / sigma

        if abs(xi) < 1e-6:
            # Caso Gumbel (limite xi→0)
            ll = -n * np.log(sigma) - np.sum(z) - np.sum(np.exp(-z))
        else:
            t = 1 + xi * z
            if np.any(t <= 0):
                return 1e10
            ll = (-n * np.log(sigma)
                  - (1 + 1/xi) * np.sum(np.log(t))
                  - np.sum(t**(-1/xi)))

        return -ll  # negativo para minimização

    
    # Fit único ativo


    def fit_single(
        self,
        series: pd.Series,
        ticker: Optional[str] = None,
        block_size: int = 22,
        use_minima: bool = True,
    ) -> GEVResult:
        """
        Extrai block maxima e ajusta GEV.

        """
        ticker = ticker or str(series.name or "ativo")

        # Extrair block maxima
        block_maxima = extract_block_maxima(series, block_size, use_minima)
        data = block_maxima.values
        n = len(data)

        if n < 10:
            raise ValueError(f"{ticker}: apenas {n} block maxima. Insuficiente para GEV.")

        # Estimativas iniciais via método dos momentos
        mu0    = np.mean(data)
        sigma0 = np.std(data) * np.sqrt(6) / np.pi
        xi0    = 0.1

        x0 = np.array([xi0, mu0, sigma0])
        bounds = [(-2.0, 2.0), (None, None), (1e-8, None)]

        result = minimize(
            self._gev_log_likelihood,
            x0,
            args=(data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        xi, mu, sigma = result.x
        ll = -result.fun
        k = 3
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n)

        gev_result = GEVResult(
            ticker=ticker,
            xi=xi, mu=mu, sigma=sigma,
            n_blocks=n,
            block_size=block_size,
            log_likelihood=ll,
            aic=aic, bic=bic,
            converged=result.success,
            block_maxima=data,
        )

        logger.info(
            f"{ticker:15s}  GEV  {gev_result.gev_type}  "
            f"mu={mu:.4f} sigma={sigma:.4f}  "
            f"AIC={aic:.2f}"
        )
        return gev_result

    # Fit todos os ativos

    def fit_all(
        self,
        returns_df: pd.DataFrame,
        block_size: int = 22,
        use_minima: bool = True,
    ) -> Dict[str, GEVResult]:
        """
        Ajusta GEV para todos os ativos.
        """
        logger.info(
            f"Ajustando GEV para {len(returns_df.columns)} ativos  "
            f"block_size={block_size}"
        )

        for ticker in returns_df.columns:
            series = returns_df[ticker].dropna()
            try:
                result = self.fit_single(series, ticker, block_size, use_minima)
                self.results_[ticker] = result
            except Exception as e:
                logger.error(f"{ticker}: GEV falhou — {e}")

        self._is_fitted = True
        return self.results_

    # Return Levels


    def get_return_levels(
        self,
        return_periods: List[float] = [10, 20, 50, 100],
        block_size: int = 22,
    ) -> pd.DataFrame:
        """
        Calcula return levels para múltiplos períodos e ativos.

        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")

        # Converter anos 
        blocks_per_year = 252 / block_size  # ≈ 11.5 para bloco mensal
        rows = {}

        for ticker, result in self.results_.items():
            row = {}
            for years in return_periods:
                n_blocks = years * blocks_per_year
                rl = result.return_level(n_blocks)
                row[f"{years}Y"] = rl
            rows[ticker] = row

        df = pd.DataFrame(rows).T
        df.index.name = "ticker"
        return df

    def get_params_df(self) -> pd.DataFrame:
        """Retorna parâmetros GEV estimados por ativo."""
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        rows = []
        for ticker, r in self.results_.items():
            rows.append({
                "ticker": ticker,
                "xi": r.xi,
                "mu": r.mu,
                "sigma": r.sigma,
                "gev_type": r.gev_type,
                "n_blocks": r.n_blocks,
                "aic": r.aic,
                "bic": r.bic,
                "converged": r.converged,
            })
        return pd.DataFrame(rows).set_index("ticker")

    def ks_test(self) -> pd.DataFrame:
        """
        Teste Kolmogorov-Smirnov de aderência GEV para cada ativo.

        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() primeiro.")
        rows = []
        for ticker, r in self.results_.items():
            data = r.block_maxima
            # CDF GEV
            if abs(r.xi) < 1e-6:
                cdf_vals = np.exp(-np.exp(-(data - r.mu) / r.sigma))
            else:
                t = 1 + r.xi * (data - r.mu) / r.sigma
                t = np.maximum(t, 1e-10)
                cdf_vals = np.exp(-t**(-1/r.xi))

            ks_stat, ks_pval = stats.kstest(data, lambda x: np.interp(
                x, np.sort(data), np.sort(cdf_vals)
            ))
            rows.append({
                "ticker": ticker,
                "ks_statistic": ks_stat,
                "p_value": ks_pval,
                "passed": ks_pval > 0.05,
            })
        return pd.DataFrame(rows).set_index("ticker")
    

def fit_gev_all(
    returns_df: pd.DataFrame,
    block_size: int = 22,
    return_periods: List[float] = [5, 10, 20, 50],
    save_path: Optional[str] = None,
) -> Tuple[GEVFitter, pd.DataFrame, pd.DataFrame]:
    """
    Wrapper para main_pipeline.py

    """
    fitter = GEVFitter()
    fitter.fit_all(returns_df, block_size=block_size)

    params_df     = fitter.get_params_df()
    return_levels = fitter.get_return_levels(return_periods, block_size)

    if save_path:
        from pathlib import Path
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(p / "gev_params.csv")
        return_levels.to_csv(p / "gev_return_levels.csv")
        logger.info(f"GEV outputs salvos em {save_path}")

    return fitter, params_df, return_levels

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

  
    np.random.seed(42)
    T = 1500
    n = 4
    dates = pd.date_range("2018-01-01", periods=T, freq="B")

    # Simular retornos com caudas pesadas (t-Student)
    returns_sim = stats.t.rvs(df=5, size=(T, n)) * 0.01
    returns_df = pd.DataFrame(
        returns_sim,
        index=dates,
        columns=["PETR4", "VALE3", "ITUB4", "BBAS3"]
    )

    print(f"returns_df: {returns_df.shape}")

    # Block maxima (mensal)
    print("\n--- Block Maxima mensais ---")
    bm = extract_block_maxima(returns_df["PETR4"], block_size=22, use_minima=True)
    print(f"N blocos: {len(bm)}  min={bm.min():.4f}  max={bm.max():.4f}")

    # GEV fit
    print("\n--- GEVFitter.fit_all() ---")
    fitter, params_df, return_levels = fit_gev_all(
        returns_df, block_size=22, return_periods=[5, 10, 20, 50]
    )

    print(f"\nParâmetros GEV:\n{params_df[['xi','mu','sigma','gev_type']].round(4)}")
    print(f"\nReturn Levels:\n{return_levels.round(4)}")

    # Interpretação
    print("\nInterpretação dos return levels (perdas máximas esperadas):")
    for ticker in return_levels.index:
        rl_10y = return_levels.loc[ticker, "10Y"]
        print(f"{ticker}: perda máxima esperada em 10 anos = {abs(rl_10y)*100:.2f}%")

    # KS test
    print(f"\nKS Test de aderência:\n{fitter.ks_test()}")

    print("\nTeste concluído.")
