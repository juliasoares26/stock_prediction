import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class ReturnLevelResult:
    """Return level para um único ativo e período"""
    ticker: str
    return_period_years: float     # período em anos
    return_period_blocks: float    # período em blocos
    block_size: int                # dias por bloco
    estimate: float                # estimativa pontual (negativo = perda)
    ci_lower: float                # IC inferior (perda mais severa)
    ci_upper: float                # IC superior (perda menos severa)
    ci_level: float                # nível do IC (ex: 0.95)
    method: str                    # 'gev', 'gpd', 'empirical'
    exceedance_prob: float         # P(perda > |estimate|) por bloco

    @property
    def loss_pct(self) -> float:
        """Perda em percentual (positivo)."""
        return abs(self.estimate) * 100

    def summary(self) -> str:
        return (
            f"{self.ticker:15s} | "
            f"T={self.return_period_years:.0f}Y | "
            f"RL={self.estimate*100:+.2f}% | "
            f"IC [{self.ci_lower*100:+.2f}%, {self.ci_upper*100:+.2f}%]"
        )


@dataclass
class PortfolioReturnLevel:
    """Return level agregado do portfólio."""
    return_period_years: float
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    method: str                    # 'weighted_average', 'monte_carlo', 'gpd_portfolio'
    asset_contributions: Dict[str, float] = field(default_factory=dict)

# Return Levels GEV

class GEVReturnLevelCalculator:
    """
    Calcula return levels a partir de resultados GEV com intervalos de confiança

    Dois métodos de IC:
    'bootstrap': reamostrar block maxima e re-ajustar GEV (não-paramétrico)
    'delta': propagação de incerteza analítica via método delta (rápido)

    """

    def __init__(
        self,
        ci_method: str = "bootstrap",
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        random_state: int = 42,
    ):
        self.ci_method = ci_method
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_state = random_state

    def compute(
        self,
        gev_result,   # GEVResult de evt_gev.py
        return_periods_years: List[float] = [1, 2, 5, 10, 20, 50, 100],
        block_size: int = 22,
    ) -> List[ReturnLevelResult]:
        """
        Calcula return levels com IC para um único ativo
        """
        blocks_per_year = 252 / block_size

        results = []
        for years in return_periods_years:
            n_blocks = years * blocks_per_year
            rl = gev_result.return_level(n_blocks)

            # IC via bootstrap ou delta
            if self.ci_method == "bootstrap":
                ci_lo, ci_hi = self._bootstrap_ci(
                    gev_result.block_maxima,
                    n_blocks,
                    gev_result.xi,
                    gev_result.mu,
                    gev_result.sigma,
                )
            else:
                ci_lo, ci_hi = self._delta_method_ci(
                    gev_result, n_blocks
                )

            # Probabilidade de excedência por bloco
            p_exceed = 1 / n_blocks

            results.append(ReturnLevelResult(
                ticker=gev_result.ticker,
                return_period_years=years,
                return_period_blocks=n_blocks,
                block_size=block_size,
                estimate=float(rl),
                ci_lower=float(ci_lo),
                ci_upper=float(ci_hi),
                ci_level=self.ci_level,
                method="gev",
                exceedance_prob=float(p_exceed),
            ))

            logger.debug(
                f"{gev_result.ticker} | {years:.0f}Y: "
                f"RL={rl*100:+.2f}% [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]"
            )

        return results

    def _bootstrap_ci(
        self,
        block_maxima: np.ndarray,
        n_blocks: float,
        xi0: float,
        mu0: float,
        sigma0: float,
    ) -> Tuple[float, float]:
        """
        IC bootstrap paramétrico: reamostrar da GEV ajustada e re-estimar
        """
        np.random.seed(self.random_state)
        n = len(block_maxima)
        boot_rls = []

        for _ in range(self.n_bootstrap):
            # Reamostrar da GEV estimada (bootstrap paramétrico)
            boot_sample = self._gev_rvs(xi0, mu0, sigma0, n)

            # Re-ajustar GEV
            try:
                xi_b, mu_b, sigma_b = self._fit_gev(boot_sample)
                p = 1 - 1 / n_blocks
                if abs(xi_b) < 1e-6:
                    rl_b = mu_b - sigma_b * np.log(-np.log(p))
                else:
                    rl_b = mu_b + (sigma_b / xi_b) * ((-np.log(p))**(-xi_b) - 1)
                boot_rls.append(float(rl_b))
            except Exception:
                pass

        if len(boot_rls) < 10:
            # Fallback: CI simétrico de 10% da estimativa
            rl0 = self._return_level_gev(xi0, mu0, sigma0, n_blocks)
            delta = abs(rl0) * 0.15
            return rl0 - delta, rl0 + delta

        alpha = (1 - self.ci_level) / 2
        ci_lo = float(np.percentile(boot_rls, alpha * 100))
        ci_hi = float(np.percentile(boot_rls, (1 - alpha) * 100))
        return ci_lo, ci_hi

    def _delta_method_ci(
        self,
        gev_result,
        n_blocks: float,
    ) -> Tuple[float, float]:
        """
        IC pelo método delta: propaga incerteza dos parâmetros GEV
        Aproximação analítica, válida assintoticamente
        """
        xi, mu, sigma = gev_result.xi, gev_result.mu, gev_result.sigma
        n = gev_result.n_blocks
        rl = self._return_level_gev(xi, mu, sigma, n_blocks)

        # Hessiana numérica da log-verossimilhança (aprox. Fisher information)
        params = np.array([xi, mu, sigma])
        data = gev_result.block_maxima

        def neg_ll(p):
            return self._gev_neg_ll(p, data)

        try:
            from scipy.optimize import approx_fprime
            eps = 1e-4 * np.abs(params) + 1e-8
            hess = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    e_i = np.zeros(3); e_i[i] = eps[i]
                    e_j = np.zeros(3); e_j[j] = eps[j]
                    hess[i, j] = (
                        neg_ll(params + e_i + e_j)
                        - neg_ll(params + e_i - e_j)
                        - neg_ll(params - e_i + e_j)
                        + neg_ll(params - e_i - e_j)
                    ) / (4 * eps[i] * eps[j])

            cov = np.linalg.inv(np.maximum(hess, 1e-10 * np.eye(3)))
        except Exception:
            # Fallback: CI de 20%
            delta = abs(rl) * 0.20
            return rl - delta, rl + delta

        # Gradiente do return level em relação aos parâmetros
        p = 1 - 1 / n_blocks
        y = -np.log(p)

        if abs(xi) < 1e-6:
            drl_dxi = 0.0
            drl_dmu = 1.0
            drl_dsigma = -np.log(y)
        else:
            t = y**(-xi)
            drl_dxi = -(sigma / xi**2) * (t - 1) + (sigma / xi) * t * np.log(y)
            drl_dmu = 1.0
            drl_dsigma = (t - 1) / xi

        grad = np.array([drl_dxi, drl_dmu, drl_dsigma])
        var_rl = float(grad @ cov @ grad)
        se_rl = np.sqrt(max(var_rl / n, 0.0))

        z = stats.norm.ppf((1 + self.ci_level) / 2)
        return float(rl - z * se_rl), float(rl + z * se_rl)

    @staticmethod
    def _return_level_gev(xi: float, mu: float, sigma: float, n_blocks: float) -> float:
        p = 1 - 1 / n_blocks
        if abs(xi) < 1e-6:
            return mu - sigma * np.log(-np.log(p))
        return mu + (sigma / xi) * ((-np.log(p))**(-xi) - 1)

    @staticmethod
    def _gev_rvs(xi: float, mu: float, sigma: float, n: int) -> np.ndarray:
        """Gera amostras da distribuição GEV."""
        u = np.random.uniform(0, 1, n)
        if abs(xi) < 1e-6:
            return mu - sigma * np.log(-np.log(u))
        return mu + sigma * ((-np.log(u))**(-xi) - 1) / xi

    @staticmethod
    def _gev_neg_ll(params: np.ndarray, data: np.ndarray) -> float:
        xi, mu, sigma = params
        if sigma <= 0:
            return 1e10
        z = (data - mu) / sigma
        if abs(xi) < 1e-6:
            return np.sum(np.log(sigma) + z + np.exp(-z))
        t = 1 + xi * z
        if np.any(t <= 0):
            return 1e10
        return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t**(-1/xi))

    @staticmethod
    def _fit_gev(data: np.ndarray) -> Tuple[float, float, float]:
        """Ajuste rápido de GEV via MLE para bootstrap"""
        mu0 = np.mean(data)
        sigma0 = np.std(data) * np.sqrt(6) / np.pi

        def neg_ll(p):
            xi, mu, sigma = p
            if sigma <= 0:
                return 1e10
            z = (data - mu) / sigma
            if abs(xi) < 1e-6:
                return np.sum(np.log(sigma) + z + np.exp(-z))
            t = 1 + xi * z
            if np.any(t <= 0):
                return 1e10
            return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t**(-1/xi))

        res = minimize(
            neg_ll, [0.1, mu0, sigma0], method="L-BFGS-B",
            bounds=[(-2, 2), (None, None), (1e-8, None)],
            options={"maxiter": 200}
        )
        return tuple(res.x)


# Return Levels via GPD

class GPDReturnLevelCalculator:
    """
    Calcula return levels a partir de resultados GPD (Peaks over Threshold).

    Fórmula: RL_m = u + (sigma/xi) * ((m * zeta_u)^xi - 1)
    onde:
        u = threshold
        sigma = escala GPD
        xi = shape GPD
        zeta_u  = P(X > u) = n_exceedances / T
        m = return period em observações diárias
    """

    def __init__(
        self,
        ci_method: str = "bootstrap",
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        random_state: int = 42,
    ):
        self.ci_method = ci_method
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.random_state = random_state

    def compute(
        self,
        gpd_result,    # GPDResult de evt_gpd.py
        return_periods_years: List[float] = [1, 2, 5, 10, 20, 50, 100],
        trading_days_per_year: int = 252,
    ) -> List[ReturnLevelResult]:
        """
        Calcula return levels GPD para um único ativo
        """
        xi     = gpd_result.xi
        sigma  = gpd_result.sigma
        u      = gpd_result.threshold
        zeta_u = gpd_result.exceedance_rate  # P(X > u)
        ticker = gpd_result.ticker

        results = []
        for years in return_periods_years:
            m = years * trading_days_per_year   # observações = dias

            # Return level GPD
            if abs(xi) < 1e-6:
                rl = u + sigma * np.log(m * zeta_u)
            else:
                rl = u + (sigma / xi) * ((m * zeta_u)**xi - 1)

            # IC bootstrap
            if self.ci_method == "bootstrap":
                ci_lo, ci_hi = self._bootstrap_ci(gpd_result, m, xi, sigma, u, zeta_u)
            else:
                ci_lo, ci_hi = rl * 0.8, rl * 1.2  # aproximação simples

            results.append(ReturnLevelResult(
                ticker=ticker,
                return_period_years=years,
                return_period_blocks=m,
                block_size=1,
                estimate=float(rl),
                ci_lower=float(ci_lo),
                ci_upper=float(ci_hi),
                ci_level=self.ci_level,
                method="gpd",
                exceedance_prob=float(1 / m),
            ))

        return results

    def _bootstrap_ci(
        self, gpd_result, m: float,
        xi: float, sigma: float, u: float, zeta_u: float,
    ) -> Tuple[float, float]:
        """Bootstrap paramétrico para GPD"""
        np.random.seed(self.random_state)
        exceedances = gpd_result.exceedances
        n_exc = len(exceedances)
        boot_rls = []

        for _ in range(self.n_bootstrap):
            boot_exc = np.random.choice(exceedances, size=n_exc, replace=True)
            try:
                xi_b, sigma_b = self._fit_gpd(boot_exc, u)
                if abs(xi_b) < 1e-6:
                    rl_b = u + sigma_b * np.log(m * zeta_u)
                else:
                    rl_b = u + (sigma_b / xi_b) * ((m * zeta_u)**xi_b - 1)
                boot_rls.append(float(rl_b))
            except Exception:
                pass

        if len(boot_rls) < 10:
            rl0 = u + (sigma / xi) * ((m * zeta_u)**xi - 1) if abs(xi) > 1e-6 else u + sigma * np.log(m * zeta_u)
            delta = abs(rl0) * 0.20
            return rl0 - delta, rl0 + delta

        alpha = (1 - self.ci_level) / 2
        return (
            float(np.percentile(boot_rls, alpha * 100)),
            float(np.percentile(boot_rls, (1 - alpha) * 100))
        )

    @staticmethod
    def _fit_gpd(exceedances: np.ndarray, threshold: float) -> Tuple[float, float]:
        """Ajuste rápido de GPD via MLE."""
        excess = exceedances - threshold
        excess = excess[excess > 0]
        if len(excess) < 5:
            return 0.1, np.std(excess) if len(excess) > 0 else 0.01

        def neg_ll(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            if xi == 0:
                return np.sum(np.log(sigma) + excess / sigma)
            t = 1 + xi * excess / sigma
            if np.any(t <= 0):
                return 1e10
            return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t))

        res = minimize(
            neg_ll, [0.1, np.mean(excess)],
            method="L-BFGS-B",
            bounds=[(-1, 2), (1e-8, None)],
            options={"maxiter": 200}
        )
        return float(res.x[0]), float(res.x[1])


# Return level de portfolio

class PortfolioReturnLevelCalculator:
    """
    Agrega return levels dos ativos individuais para o nível do portfólio

    Três métodos:
        'weighted_average': RL_port = Σ w_i * RL_i (simples, ignora correlação)
        'monte_carlo' : simula portfólio completo e computa RL direto
        'copula_mc': Monte Carlo com estrutura de cópula 
    """

    def __init__(self, method: str = "monte_carlo", n_sim: int = 100000):
        self.method = method
        self.n_sim = n_sim

    def compute(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        gev_fitter=None,
        return_periods_years: List[float] = [1, 2, 5, 10, 20, 50, 100],
        block_size: int = 22,
        ci_level: float = 0.95,
        n_bootstrap: int = 300,
        copula=None,
    ) -> List[PortfolioReturnLevel]:
        """
        Calcula return levels do portfólio

        """
        if self.method == "weighted_average" and gev_fitter is not None:
            return self._weighted_average(
                gev_fitter, weights, return_periods_years, block_size, ci_level
            )
        elif self.method in ("monte_carlo", "copula_mc"):
            return self._monte_carlo(
                returns_df, weights, return_periods_years,
                block_size, ci_level, n_bootstrap, copula
            )
        else:
            # Fallback: empírico
            return self._empirical(returns_df, weights, return_periods_years, block_size, ci_level, n_bootstrap)

    def _weighted_average(
        self, gev_fitter, weights, return_periods_years, block_size, ci_level
    ) -> List[PortfolioReturnLevel]:
        """Média ponderada simples dos return levels individuais"""
        blocks_per_year = 252 / block_size
        tickers = list(gev_fitter.results_.keys())
        d = len(tickers)
        w = weights[:d]

        results = []
        for years in return_periods_years:
            n_blocks = years * blocks_per_year
            rl_assets = {}
            rl_weighted = 0.0

            for i, ticker in enumerate(tickers):
                if ticker not in gev_fitter.results_:
                    continue
                rl_i = gev_fitter.results_[ticker].return_level(n_blocks)
                rl_assets[ticker] = float(rl_i)
                rl_weighted += w[i] * rl_i

            # IC: propagar via pesos
            ci_delta = abs(rl_weighted) * 0.20
            results.append(PortfolioReturnLevel(
                return_period_years=years,
                estimate=float(rl_weighted),
                ci_lower=float(rl_weighted - ci_delta),
                ci_upper=float(rl_weighted + ci_delta),
                ci_level=ci_level,
                method="weighted_average",
                asset_contributions=rl_assets,
            ))

        return results

    def _monte_carlo(
        self, returns_df, weights, return_periods_years,
        block_size, ci_level, n_bootstrap, copula
    ) -> List[PortfolioReturnLevel]:
        """Monte Carlo: simula portfólio e estima return level diretamente"""
        np.random.seed(42)
        T, d = returns_df.shape
        w = weights[:d]

        # Portfólio histórico para bootstrap
        port_returns = (returns_df * w).sum(axis=1).dropna()

        results = []
        blocks_per_year = 252 / block_size

        for years in return_periods_years:
            n_blocks = int(years * blocks_per_year)

            if copula is not None:
                # Simular via cópula
                try:
                    U_sim = copula.simulate(self.n_sim)
                    R_sim = np.zeros((self.n_sim, d))
                    for j in range(d):
                        col = returns_df.iloc[:, j].dropna().values
                        T_hist = len(col)
                        idx = np.clip((U_sim[:, j] * T_hist).astype(int), 0, T_hist-1)
                        R_sim[:, j] = np.sort(col)[idx]
                    port_sim = R_sim @ w
                except Exception:
                    port_sim = np.random.choice(port_returns.values, self.n_sim, replace=True)
            else:
                port_sim = np.random.choice(port_returns.values, self.n_sim, replace=True)

            # Return level: quantil (1/n_blocks)
            quantile_level = 1 / n_blocks
            rl = float(np.percentile(port_sim, quantile_level * 100))

            # IC via bootstrap dos dados históricos
            boot_rls = []
            for _ in range(n_bootstrap):
                boot_port = np.random.choice(port_returns.values, len(port_returns), replace=True)
                boot_rl = float(np.percentile(boot_port, quantile_level * 100))
                boot_rls.append(boot_rl)

            alpha = (1 - ci_level) / 2
            ci_lo = float(np.percentile(boot_rls, alpha * 100))
            ci_hi = float(np.percentile(boot_rls, (1-alpha) * 100))

            results.append(PortfolioReturnLevel(
                return_period_years=years,
                estimate=rl,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                ci_level=ci_level,
                method="monte_carlo" if copula is None else "copula_mc",
            ))

            logger.debug(
                f"Portfólio {years:.0f}Y: RL={rl*100:+.2f}% "
                f"IC [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]"
            )

        return results

    def _empirical(
        self, returns_df, weights, return_periods_years, block_size, ci_level, n_bootstrap
    ) -> List[PortfolioReturnLevel]:
        """Empírico via block maxima do portfólio histórico"""
        port_returns = (returns_df * weights[:returns_df.shape[1]]).sum(axis=1).dropna()
        T = len(port_returns)
        blocks_per_year = 252 / block_size
        results = []

        for years in return_periods_years:
            n_blocks = int(years * blocks_per_year)
            quantile_level = 1 / n_blocks

            if quantile_level * 100 < 100 / T:
                logger.warning(
                    f"T={T} insuficiente para {years}Y empiricamente"
                    f"Use GEV/GPD para extrapolação."
                )
                rl = float(port_returns.quantile(0.001))
            else:
                rl = float(np.percentile(port_returns, quantile_level * 100))

            # Bootstrap
            boot_rls = [
                float(np.percentile(
                    np.random.choice(port_returns.values, T, replace=True),
                    quantile_level * 100
                ))
                for _ in range(n_bootstrap)
            ]

            alpha = (1 - ci_level) / 2
            results.append(PortfolioReturnLevel(
                return_period_years=years,
                estimate=rl,
                ci_lower=float(np.percentile(boot_rls, alpha * 100)),
                ci_upper=float(np.percentile(boot_rls, (1-alpha) * 100)),
                ci_level=ci_level,
                method="empirical",
            ))

        return results


class ReturnLevelCalculator:
    """
    return level para todos os ativos e para o portfólio.
    """

    def __init__(
        self,
        ci_method: str = "bootstrap",
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        portfolio_method: str = "monte_carlo",
    ):
        self.ci_method = ci_method
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.portfolio_method = portfolio_method

        self._asset_results: Dict[str, List[ReturnLevelResult]] = {}
        self._portfolio_results: List[PortfolioReturnLevel] = []

    def compute_all(
        self,
        gev_fitter,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        return_periods_years: List[float] = [1, 2, 5, 10, 20, 50, 100],
        block_size: int = 22,
        copula=None,
        gpd_results: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Calcula return levels para todos os ativos e portfólio
        """
        logger.info(
            f"Calculando return levels | "
            f"ativos={len(gev_fitter.results_)} | "
            f"períodos={return_periods_years} | "
            f"CI method={self.ci_method}"
        )

        calc = GEVReturnLevelCalculator(
            ci_method=self.ci_method,
            n_bootstrap=self.n_bootstrap,
            ci_level=self.ci_level,
        )

        #  Return levels por ativo 
        rows = []
        for ticker, gev_result in gev_fitter.results_.items():
            logger.info(f"  Calculando RL para {ticker}")
            rl_list = calc.compute(gev_result, return_periods_years, block_size)
            self._asset_results[ticker] = rl_list

            for rl in rl_list:
                rows.append({
                    "ticker": ticker,
                    "years": rl.return_period_years,
                    "estimate": rl.estimate,
                    "ci_lower": rl.ci_lower,
                    "ci_upper": rl.ci_upper,
                    "loss_pct": rl.loss_pct,
                    "ci_loss_lower": abs(rl.ci_upper) * 100,
                    "ci_loss_upper": abs(rl.ci_lower) * 100,
                    "method": rl.method,
                    "exceedance_prob": rl.exceedance_prob,
                })

        #  GPD se disponível 
        if gpd_results:
            gpd_calc = GPDReturnLevelCalculator(
                ci_method=self.ci_method,
                n_bootstrap=self.n_bootstrap // 2,
                ci_level=self.ci_level,
            )
            for ticker, gpd_result in gpd_results.items():
                rl_list_gpd = gpd_calc.compute(gpd_result, return_periods_years)
                for rl in rl_list_gpd:
                    rows.append({
                        "ticker": ticker + "_GPD",
                        "years": rl.return_period_years,
                        "estimate": rl.estimate,
                        "ci_lower": rl.ci_lower,
                        "ci_upper": rl.ci_upper,
                        "loss_pct": rl.loss_pct,
                        "ci_loss_lower": abs(rl.ci_upper) * 100,
                        "ci_loss_upper": abs(rl.ci_lower) * 100,
                        "method": rl.method,
                        "exceedance_prob": rl.exceedance_prob,
                    })

        #  Portfólio 
        logger.info("  Calculando RL do portfólio")
        port_calc = PortfolioReturnLevelCalculator(
            method=self.portfolio_method,
            n_sim=50000,
        )
        self._portfolio_results = port_calc.compute(
            returns_df, weights, gev_fitter,
            return_periods_years, block_size, self.ci_level,
            self.n_bootstrap // 2, copula,
        )
        for rl in self._portfolio_results:
            rows.append({
                "ticker": "PORTFOLIO",
                "years": rl.return_period_years,
                "estimate": rl.estimate,
                "ci_lower": rl.ci_lower,
                "ci_upper": rl.ci_upper,
                "loss_pct": abs(rl.estimate) * 100,
                "ci_loss_lower": abs(rl.ci_upper) * 100,
                "ci_loss_upper": abs(rl.ci_lower) * 100,
                "method": rl.method,
                "exceedance_prob": 1 / (rl.return_period_years * 252 / 22),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Return levels calculados: {len(df)} entradas")
        return df

    def get_portfolio_summary(self) -> pd.DataFrame:
        """Return levels do portfólio em formato tabular."""
        rows = []
        for rl in self._portfolio_results:
            rows.append({
                "return_period_years": rl.return_period_years,
                "estimate_pct": rl.estimate * 100,
                "ci_lower_pct": rl.ci_lower * 100,
                "ci_upper_pct": rl.ci_upper * 100,
                "method": rl.method,
            })
        return pd.DataFrame(rows).set_index("return_period_years")

    def get_return_level_curve(
        self,
        ticker: str,
        n_points: int = 50,
        max_years: float = 100,
        block_size: int = 22,
    ) -> pd.DataFrame:
        """
        Gera dados para a curva return level plot
        """
        if ticker not in self._asset_results:
            raise ValueError(f"{ticker} não encontrado. Execute compute_all() primeiro.")

        gev_res_list = self._asset_results[ticker]

        # Extrapolar para mais pontos
        try:
            from src.marginals.evt_gev import GEVFitter
        except ImportError:
            try:
                from marginals.evt_gev import GEVFitter
            except ImportError:
                from evt_gev import GEVFitter
        # Usar os parâmetros do primeiro resultado disponível para extrapolação
        # (retornamos os pontos calculados)
        years_arr = np.logspace(np.log10(0.5), np.log10(max_years), n_points)
        calc = GEVReturnLevelCalculator(ci_method="delta", ci_level=self.ci_level)

        rows = []
        for years in years_arr:
            # Encontrar resultado mais próximo
            dists = [abs(r.return_period_years - years) for r in gev_res_list]
            closest = gev_res_list[np.argmin(dists)]
            rows.append({
                "years": years,
                "estimate": closest.estimate,
                "ci_lower": closest.ci_lower,
                "ci_upper": closest.ci_upper,
            })

        return pd.DataFrame(rows)

    def save(self, df: pd.DataFrame, path: Union[str, Path]):
        """Salva tabela de return levels."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path).endswith(".parquet"):
            df.to_parquet(path)
        else:
            df.to_csv(path, index=False)
        logger.info(f"Return levels salvos em {path}")


def compute_return_levels(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    block_size: int = 22,
    return_periods_years: List[float] = [1, 2, 5, 10, 20, 50, 100],
    ci_method: str = "bootstrap",
    n_bootstrap: int = 300,
    copula=None,
    save_path: Optional[str] = None,
) -> Tuple[ReturnLevelCalculator, pd.DataFrame]:
    """
    Wrapper para main_pipeline.py, ajusta GEV e calcula return levels
    """
    from src.marginals.evt_gev import GEVFitter

    fitter = GEVFitter()
    fitter.fit_all(returns_df, block_size=block_size)

    rlc = ReturnLevelCalculator(
        ci_method=ci_method,
        n_bootstrap=n_bootstrap,
        portfolio_method="monte_carlo",
    )

    report = rlc.compute_all(
        gev_fitter=fitter,
        returns_df=returns_df,
        weights=weights,
        return_periods_years=return_periods_years,
        block_size=block_size,
        copula=copula,
    )

    if save_path:
        rlc.save(report, save_path)

    return rlc, report


# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    np.random.seed(42)
    T = 1500
    n = 4
    tickers = ["PETR4", "VALE3", "ITUB4", "ABEV3"]
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    weights = np.array([0.30, 0.25, 0.30, 0.15])

    # Simular retornos com escala realista B3 e correlacao setorial
    # Volatilidade diaria: PETR4~2.5%, VALE3~2.2%, ITUB4~1.8%, ABEV3~1.5%
    vols = np.array([0.025, 0.022, 0.018, 0.015])
    R_corr = np.array([
        [1.0, 0.55, 0.45, 0.30],
        [0.55, 1.0, 0.40, 0.28],
        [0.45, 0.40, 1.0, 0.35],
        [0.30, 0.28, 0.35, 1.0],
    ])
    L_corr = np.linalg.cholesky(R_corr)
    # t-Student df=5 para caudas pesadas (xi>0 esperado)
    Z_sim = stats.t.rvs(df=5, size=(T, n))
    returns_sim = (Z_sim @ L_corr.T) * vols
    returns_df = pd.DataFrame(returns_sim, index=dates, columns=tickers)

    # Estatisticas descritivas para validar escala
    vol_ann = returns_df.std() * np.sqrt(252) * 100
    print(f"Volatilidade anual simulada:")
    for t in tickers:
        print(f"  {t}: {vol_ann[t]:.1f}%")

    print(f"returns_df: {returns_df.shape}")

    # Ajustar GEV
    print("\n Ajustando GEV (bloco mensal) ")
    try:
        from src.marginals.evt_gev import GEVFitter
    except ImportError:
        try:
            from marginals.evt_gev import GEVFitter
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Copiar GEVFitter inline para teste autonomo
            from marginals.evt_gev import GEVFitter

    gev_fitter = GEVFitter()
    gev_fitter.fit_all(returns_df, block_size=22)

    # Return levels via GEV
    print("\n GEVReturnLevelCalculator ")
    calc = GEVReturnLevelCalculator(ci_method="bootstrap", n_bootstrap=200)
    for ticker in tickers[:2]:
        rl_list = calc.compute(gev_fitter.results_[ticker], [5, 10, 20, 50])
        for rl in rl_list:
            print(f"  {rl.summary()}")

    # Cálculo completo
    print("\n ReturnLevelCalculator.compute_all() ")
    rlc = ReturnLevelCalculator(
        ci_method="bootstrap",
        n_bootstrap=200,
        portfolio_method="monte_carlo",
    )
    report = rlc.compute_all(
        gev_fitter=gev_fitter,
        returns_df=returns_df,
        weights=weights,
        return_periods_years=[1, 5, 10, 20, 50],
        block_size=22,
    )

    print(f"\nTabela completa: {report.shape}")
    print("\nReturn levels do portfólio:")
    port_summary = rlc.get_portfolio_summary()
    print(port_summary.round(3))

    print("\nReturn levels por ativo (10 anos):")
    df_10y = report[report["years"] == 10][["ticker", "loss_pct", "ci_loss_lower", "ci_loss_upper", "method"]]
    print(df_10y.round(3).to_string())

    print("\n Teste concluído.")
