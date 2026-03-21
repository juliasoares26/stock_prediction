
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Define um cenário de stress"""
    name: str
    description: str
    shock_type: str          # 'historical', 'hypothetical', 'evt', 'correlation'
    # Para cenários históricos: janela de datas
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # Para cenários hipotéticos: choques por ativo
    shocks: Optional[Dict[str, float]] = None    # {ticker: retorno_diário_choque}
    vol_multiplier: float = 1.0                   # multiplicador da volatilidade
    corr_multiplier: float = 1.0                  # multiplicador das correlações
    # Para cenários EVT
    quantile: float = 0.001                       # quantil extremo (ex: 0.1%)
    n_sim: int = 50000


@dataclass
class StressResult:
    """Resultado de um cenário de stress"""
    scenario_name: str
    scenario_type: str
    portfolio_return: float          # retorno total do portfólio no cenário
    portfolio_loss_pct: float        # perda em % do portfólio
    var_1pct: float                  # VaR 1% no cenário
    cvar_1pct: float                 # CVaR 1% no cenário
    worst_asset: str                 # ativo com pior desempenho
    worst_asset_return: float        # retorno do pior ativo
    asset_returns: Dict[str, float]  # retornos individuais no cenário
    description: str
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"Cenário: {self.scenario_name} ({self.scenario_type})",
            f"{'='*55}",
            f"Retorno do portfólio : {self.portfolio_return*100:+.2f}%",
            f"Perda total          : {self.portfolio_loss_pct*100:.2f}%",
            f"VaR 1%               : {self.var_1pct*100:.2f}%",
            f"CVaR 1%              : {self.cvar_1pct*100:.2f}%",
            f"Pior ativo           : {self.worst_asset} ({self.worst_asset_return*100:+.2f}%)",
            f"Descrição            : {self.description}",
        ]
        return "\n".join(lines)


# cenários históricos

# Episódios de crise relevantes para B3
HISTORICAL_CRISES = {
    "gfc_2008": StressScenario(
        name="Crise Financeira Global 2008",
        description="Colapso do Lehman Brothers e crise do subprime",
        shock_type="historical",
        start_date="2008-09-01",
        end_date="2009-03-31",
    ),
    "covid_2020": StressScenario(
        name="COVID-19 (Mar 2020)",
        description="Crash de março de 2020 — pior mês do Ibovespa em décadas",
        shock_type="historical",
        start_date="2020-02-20",
        end_date="2020-03-23",
    ),
    "lula_eleicao_2002": StressScenario(
        name="Eleição Lula 2002",
        description="Crise de confiança pré-eleição presidencial",
        shock_type="historical",
        start_date="2002-05-01",
        end_date="2002-10-31",
    ),
    "dilma_impeachment_2015": StressScenario(
        name="Crise Política 2015-2016",
        description="Impeachment e recessão brasileira",
        shock_type="historical",
        start_date="2015-01-01",
        end_date="2016-06-30",
    ),
    "temer_jbs_2017": StressScenario(
        name="Crise Temer/JBS 2017",
        description="Gravação e crise política de maio de 2017",
        shock_type="historical",
        start_date="2017-05-17",
        end_date="2017-06-30",
    ),
    "crise_cambio_2018": StressScenario(
        name="Crise Cambial 2018",
        description="Desvalorização do Real e truckers strike",
        shock_type="historical",
        start_date="2018-05-01",
        end_date="2018-07-31",
    ),
    "petro_crash_2020": StressScenario(
        name="Crash do Petróleo (Mar 2020)",
        description="Guerra de preços OPEC+ durante COVID",
        shock_type="historical",
        start_date="2020-03-09",
        end_date="2020-04-20",
    ),
}

# Cenários hipotéticos calibrados
HYPOTHETICAL_SCENARIOS = {
    "crash_setorial_financeiro": StressScenario(
        name="Crash Setor Financeiro",
        description="Colapso bancário sistêmico: -25% em bancos, -10% demais",
        shock_type="hypothetical",
        shocks={
            "ITUB4.SA": -0.25, "BBAS3.SA": -0.25, "BBDC4.SA": -0.25,
            "B3SA3.SA": -0.20, "SANB11.SA": -0.25,
        },
        vol_multiplier=2.5,
    ),
    "crash_commodities": StressScenario(
        name="Crash de Commodities",
        description="Colapso de metais e petróleo: -30% em commodities",
        shock_type="hypothetical",
        shocks={
            "PETR4.SA": -0.30, "VALE3.SA": -0.30,
            "SUZB3.SA": -0.20, "GGBR4.SA": -0.25,
        },
        vol_multiplier=2.0,
    ),
    "desvalorizacao_brl": StressScenario(
        name="Desvalorização Extrema BRL",
        description="BRL/USD passa de 5 para 8: impacto diferenciado por setor",
        shock_type="hypothetical",
        shocks={
            "PETR4.SA": +0.05,   # exportadora: beneficia
            "VALE3.SA": +0.08,   # exportadora: beneficia
            "LREN3.SA": -0.15,   # varejista: sofre com custos
            "ABEV3.SA": -0.10,   # insumos importados
        },
        vol_multiplier=1.8,
    ),
    "black_swan": StressScenario(
        name="Black Swan Sistêmico",
        description="Choque simultâneo de -3 sigmas em todos os ativos",
        shock_type="hypothetical",
        shocks={},
        vol_multiplier=3.0,
        corr_multiplier=1.5,   # correlações aumentam em crises
    ),
    "stress_correlation": StressScenario(
        name="Stress de Correlação",
        description="Correlações convergem para 0.9 (flight to quality)",
        shock_type="correlation",
        corr_multiplier=2.0,
        vol_multiplier=2.0,
    ),
}


class StressTester:
    """
    Motor de stress testing para portfólios de ações brasileiras.

    Integra resultados de GARCH, GEV, cópulas e regimes para gerar
    cenários adversos e medir o impacto no portfólio.

    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        garch_results: Optional[Dict] = None,
        gev_fitter=None,
        copula=None,
        regime_labels: Optional[pd.Series] = None,
    ):
        self.returns_df = returns_df
        self.weights = np.array(weights)
        self.garch_results = garch_results or {}
        self.gev_fitter = gev_fitter
        self.copula = copula
        self.regime_labels = regime_labels
        self.tickers = list(returns_df.columns)
        self.n = len(self.tickers)

        # Portfólio igualmente ponderado como fallback
        if len(self.weights) == 0:
            self.weights = np.ones(self.n) / self.n

        # Cache de estatísticas
        self._port_returns = (returns_df * self.weights).sum(axis=1)
        self._annual_vol = float(self._port_returns.std() * np.sqrt(252))
        self._results: List[StressResult] = []

    # Cenário historico

    def run_historical(
        self,
        scenario: StressScenario,
    ) -> Optional[StressResult]:
        """
        Replica um período histórico real

        Calcula retornos cumulativos de cada ativo durante o episódio e aplica nos pesos atuais do portfólio

        """
        start = scenario.start_date
        end   = scenario.end_date

        # Filtrar dados do período
        mask = (self.returns_df.index >= start) & (self.returns_df.index <= end)
        period_df = self.returns_df[mask]

        if len(period_df) == 0:
            logger.warning(
                f"Cenário '{scenario.name}': período {start}–{end} "
                f"não encontrado nos dados (mínimo={self.returns_df.index.min().date()}, "
                f"máximo={self.returns_df.index.max().date()}). Pulando."
            )
            return None

        # Retornos cumulativos: (1 + r_1)(1 + r_2)... - 1
        cum_returns = {}
        for ticker in self.tickers:
            if ticker in period_df.columns:
                r = period_df[ticker].dropna()
                if len(r) > 0:
                    # Log-returns: soma = retorno cumulativo log
                    cum_returns[ticker] = float(r.sum())
                else:
                    cum_returns[ticker] = 0.0
            else:
                cum_returns[ticker] = 0.0

        # P&L do portfólio
        port_return = sum(
            self.weights[i] * cum_returns.get(t, 0.0)
            for i, t in enumerate(self.tickers)
        )

        # Distribuição de retornos durante o período para VaR/CVaR
        port_daily = (period_df[self.tickers] * self.weights).sum(axis=1).dropna()
        var_1pct  = float(np.percentile(port_daily, 1)) if len(port_daily) > 0 else port_return
        cvar_1pct = float(port_daily[port_daily <= var_1pct].mean()) if len(port_daily) > 0 else port_return

        worst_asset = min(cum_returns, key=cum_returns.get)

        result = StressResult(
            scenario_name=scenario.name,
            scenario_type="historical",
            portfolio_return=port_return,
            portfolio_loss_pct=max(-port_return, 0.0),
            var_1pct=var_1pct,
            cvar_1pct=cvar_1pct,
            worst_asset=worst_asset,
            worst_asset_return=cum_returns[worst_asset],
            asset_returns=cum_returns,
            description=scenario.description,
            metadata={
                "start_date": start,
                "end_date": end,
                "n_trading_days": len(period_df),
            },
        )
        logger.info(f"[histórico] {scenario.name}: portfólio {port_return*100:+.2f}%")
        return result

   # Cenário hipotetico

    def run_hypothetical(
        self,
        scenario: StressScenario,
    ) -> StressResult:
        """
        Aplica choques manuais calibrados e propaga via correlações históricas

        Para ativos sem choque explícito, propaga impacto via correlações:
            r_j = rho(i,j) * choque_i + epsilon_j
        onde epsilon_j ~ N(0, vol_j * vol_multiplier).

        """
        shocks = scenario.shocks or {}
        vol_mult  = scenario.vol_multiplier
        corr_mult = min(scenario.corr_multiplier, 1.5)  # cap em 1.5 para correlações válidas

        # Estatísticas históricas
        vols = self.returns_df.std().values
        corr_matrix = self.returns_df.corr().values
        corr_matrix = np.clip(corr_matrix * corr_mult, -0.999, 0.999)
        np.fill_diagonal(corr_matrix, 1.0)

        # Projetar correlações válidas (PSD)
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

        asset_returns = {}

        # Aplicar choques explícitos
        for ticker in self.tickers:
            if ticker in shocks:
                asset_returns[ticker] = float(shocks[ticker])

        # Propagar via correlações para ativos sem choque
        shocked_tickers = list(shocks.keys())
        for j, ticker in enumerate(self.tickers):
            if ticker in asset_returns:
                continue

            vol_j = float(vols[j]) * vol_mult
            propagated = 0.0
            n_propagated = 0

            for i, shocked_t in enumerate(shocked_tickers):
                if shocked_t in self.tickers:
                    idx_i = self.tickers.index(shocked_t)
                    rho_ij = corr_matrix[j, idx_i]
                    propagated += rho_ij * shocks[shocked_t]
                    n_propagated += 1

            if n_propagated > 0:
                propagated /= n_propagated

            # Adicionar ruído idiossincrático
            eps = np.random.normal(0, vol_j * 0.3)
            asset_returns[ticker] = propagated + eps

        # Black Swan: -3 sigma em todos
        if scenario.name == "Black Swan Sistêmico" or vol_mult >= 3.0:
            for j, ticker in enumerate(self.tickers):
                if ticker not in shocks:
                    asset_returns[ticker] = -3.0 * float(vols[j]) * vol_mult

        # P&L do portfólio
        port_return = sum(
            self.weights[i] * asset_returns.get(t, 0.0)
            for i, t in enumerate(self.tickers)
        )

        # VaR/CVaR via simulação Monte Carlo do cenário
        port_sim = self._simulate_stressed_portfolio(
            n_sim=10000,
            vol_multiplier=vol_mult,
            corr_matrix=corr_matrix,
            base_returns={t: asset_returns.get(t, 0.0) for t in self.tickers},
        )
        var_1pct  = float(np.percentile(port_sim, 1))
        cvar_1pct = float(np.mean(port_sim[port_sim <= var_1pct]))

        worst_asset = min(asset_returns, key=asset_returns.get)

        result = StressResult(
            scenario_name=scenario.name,
            scenario_type="hypothetical",
            portfolio_return=port_return,
            portfolio_loss_pct=max(-port_return, 0.0),
            var_1pct=var_1pct,
            cvar_1pct=cvar_1pct,
            worst_asset=worst_asset,
            worst_asset_return=asset_returns[worst_asset],
            asset_returns=asset_returns,
            description=scenario.description,
            metadata={"vol_multiplier": vol_mult, "corr_multiplier": corr_mult},
        )
        logger.info(f"[hipotético] {scenario.name}: portfólio {port_return*100:+.2f}%")
        return result

    def _simulate_stressed_portfolio(
        self,
        n_sim: int,
        vol_multiplier: float,
        corr_matrix: np.ndarray,
        base_returns: Dict[str, float],
    ) -> np.ndarray:
        """Simula retornos do portfólio sob condições de stress"""
        vols = self.returns_df.std().values * vol_multiplier

        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            L = np.eye(self.n)

        base = np.array([base_returns.get(t, 0.0) for t in self.tickers])
        Z = np.random.standard_normal((n_sim, self.n))
        R = base + Z @ L.T * vols
        return R @ self.weights

    #evt

    def run_evt_scenario(
        self,
        quantile: float = 0.001,
        use_copula: bool = True,
        n_sim: int = 100000,
        scenario_name: str = "EVT Tail Scenario",
    ) -> StressResult:
        """
        Gera cenário de perda extrema usando EVT e estrutura de cópula

        Se cópula disponível: simula uniformes correlacionadas e transforma via quantis da distribuição marginal
        Extrai o quantil `quantile` do portfólio simulado
        Retorna o cenário correspondente à perda extrema

        """
        logger.info(
            f"Gerando cenário EVT  quantil={quantile:.4f}  "
            f"cópula={'sim' if use_copula and self.copula else 'não'}"
        )

        if use_copula and self.copula is not None:
            # Simular via cópula
            try:
                U_sim = self.copula.simulate(n_sim)
            except Exception as e:
                logger.warning(f"Cópula falhou: {e}. Usando correlação histórica.")
                U_sim = self._simulate_gaussian_copula(n_sim)
        else:
            U_sim = self._simulate_gaussian_copula(n_sim)

        # Transformar uniformes para retornos via quantis empíricos
        R_sim = np.zeros((n_sim, self.n))
        for j, ticker in enumerate(self.tickers):
            empirical_quantiles = np.sort(self.returns_df[ticker].dropna().values)
            T_hist = len(empirical_quantiles)
            # Interpolação: U_sim[:, j] → retorno via quantil histórico
            idx = np.clip((U_sim[:, j] * T_hist).astype(int), 0, T_hist - 1)
            R_sim[:, j] = empirical_quantiles[idx]

        # Retorno do portfólio
        port_sim = R_sim @ self.weights

        # Cenário no quantil extremo
        port_quantile = float(np.percentile(port_sim, quantile * 100))

        # Selecionar a simulação mais próxima do quantil
        dists = np.abs(port_sim - port_quantile)
        best_idx = int(np.argmin(dists))
        scenario_returns = {t: float(R_sim[best_idx, j]) for j, t in enumerate(self.tickers)}

        # VaR e CVaR
        var_1pct  = float(np.percentile(port_sim, 1))
        cvar_1pct = float(np.mean(port_sim[port_sim <= var_1pct]))

        worst_asset = min(scenario_returns, key=scenario_returns.get)

        result = StressResult(
            scenario_name=scenario_name,
            scenario_type="evt",
            portfolio_return=port_quantile,
            portfolio_loss_pct=max(-port_quantile, 0.0),
            var_1pct=var_1pct,
            cvar_1pct=cvar_1pct,
            worst_asset=worst_asset,
            worst_asset_return=scenario_returns[worst_asset],
            asset_returns=scenario_returns,
            description=f"Cenário EVT no quantil {quantile:.4f} ({1/quantile:.0f}-year event)",
            metadata={
                "quantile": quantile,
                "n_sim": n_sim,
                "used_copula": use_copula and self.copula is not None,
            },
        )
        logger.info(
            f"[EVT] {scenario_name}: portfólio {port_quantile*100:+.2f}% "
            f"(1-em-{1/quantile:.0f} evento)"
        )
        return result

    def _simulate_gaussian_copula(self, n_sim: int) -> np.ndarray:
        """Fallback: simula uniformes via cópula Gaussiana histórica"""
        R = self.returns_df.corr().values
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-6)
        R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(R_psd)
        Z = np.random.standard_normal((n_sim, self.n)) @ L.T
        return stats.norm.cdf(Z).astype(np.float32)

    # Cenário de stress de correlação

    def run_correlation_stress(
        self,
        target_correlation: float = 0.85,
        vol_multiplier: float = 2.0,
        n_sim: int = 50000,
        scenario_name: str = "Correlation Stress",
    ) -> StressResult:
        """
        Stress onde todas as correlações convergem para um valor alto
       
        Modela o fenômeno observado durante crises onde diversificação
        falha e correlações se aproximam de 1.

        """
        d = self.n

        # Matriz de correlação estressada: lerp entre identidade e ones
        corr_stressed = (
            target_correlation * np.ones((d, d))
            + (1 - target_correlation) * np.eye(d)
        )

        # Retorno base: média histórica negativa (bear market)
        mean_ret = self.returns_df.mean().values
        vols_stressed = self.returns_df.std().values * vol_multiplier

        # Bear market: returns deslocados para baixo em 2 sigmas
        # vol_multiplier escala com target_correlation para diferenciar cenarios
        # ρ=0.7 → vol_mult_adj=2.2, ρ=0.85 → 2.35, ρ=0.95 → 2.45
        vol_mult_adj = vol_multiplier * (1.0 + 0.5 * (target_correlation - 0.5))
        vols_adj = self.returns_df.std().values * vol_mult_adj
        base_ret = mean_ret - 2.0 * vols_adj

        result = self.run_hypothetical(StressScenario(
            name=scenario_name,
            description=f"Correlações → {target_correlation:.0%}, vol × {vol_mult_adj:.2f}",
            shock_type="correlation",
            shocks={t: float(base_ret[i]) for i, t in enumerate(self.tickers)},
            vol_multiplier=vol_multiplier,
            corr_multiplier=target_correlation / max(
                self.returns_df.corr().values[np.triu_indices(d, k=1)].mean(), 0.01
            ),
        ))
        result.scenario_type = "correlation"
        return result

    # Stress por regime
  

    def run_regime_stress(
        self,
        regime: int = 1,
        quantile: float = 0.05,
    ) -> Optional[StressResult]:
        """
        Stress baseado nos retornos observados durante um regime específico
        Usa os piores `quantile` % dos retornos naquele regime

        """
        if self.regime_labels is None:
            logger.warning("regime_labels não disponível. Pulando cenário de regime.")
            return None

        common = self.regime_labels.dropna().index.intersection(self.returns_df.index)
        labels = self.regime_labels.loc[common].astype(int)
        mask = labels == regime

        if mask.sum() < 20:
            logger.warning(f"Regime {regime}: apenas {mask.sum()} obs. Insuficiente.")
            return None

        ret_regime = self.returns_df.loc[mask[mask].index]
        port_regime = (ret_regime * self.weights).sum(axis=1)
        q_val = float(np.percentile(port_regime, quantile * 100))

        # Encontrar o pior dia do regime
        worst_day_idx = port_regime.idxmin()
        asset_returns = ret_regime.loc[worst_day_idx].to_dict()
        worst_asset = min(asset_returns, key=asset_returns.get)

        var_1pct  = float(np.percentile(port_regime, 1))
        cvar_1pct = float(np.mean(port_regime[port_regime <= var_1pct]))

        result = StressResult(
            scenario_name=f"Regime {regime} — Pior {quantile:.0%}",
            scenario_type="regime",
            portfolio_return=q_val,
            portfolio_loss_pct=max(-q_val, 0.0),
            var_1pct=var_1pct,
            cvar_1pct=cvar_1pct,
            worst_asset=worst_asset,
            worst_asset_return=float(asset_returns[worst_asset]),
            asset_returns={k: float(v) for k, v in asset_returns.items()},
            description=f"Pior {quantile:.0%} dos retornos no regime {regime} (alta vol)",
            metadata={"regime": regime, "n_regime_obs": int(mask.sum())},
        )
        logger.info(f"[regime {regime}] P{quantile*100:.0f}: portfólio {q_val*100:+.2f}%")
        return result

    # Run all

    def run_all(
        self,
        include_historical: bool = True,
        include_hypothetical: bool = True,
        include_evt: bool = True,
        include_correlation: bool = True,
        evt_quantiles: List[float] = [0.01, 0.001, 0.0001],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Executa todos os cenários de stress e retorna relatório consolidado

        """
        results: List[StressResult] = []

        # Históricos
        if include_historical:
            logger.info("Executando cenários históricos...")
            for key, scenario in HISTORICAL_CRISES.items():
                r = self.run_historical(scenario)
                if r is not None:
                    results.append(r)

        # Hipotéticos
        if include_hypothetical:
            logger.info("Executando cenários hipotéticos...")
            for key, scenario in HYPOTHETICAL_SCENARIOS.items():
                if scenario.shock_type == "correlation":
                    continue
                np.random.seed(42)
                r = self.run_hypothetical(scenario)
                results.append(r)

        # Correlação
        if include_correlation:
            logger.info("Executando cenários de correlação...")
            for corr in [0.7, 0.85, 0.95]:
                r = self.run_correlation_stress(
                    target_correlation=corr,
                    vol_multiplier=2.0,
                    scenario_name=f"Correlação Stress (ρ={corr})",
                )
                results.append(r)

        # EVT
        if include_evt:
            logger.info("Executando cenários EVT...")
            for q in evt_quantiles:
                np.random.seed(42)
                r = self.run_evt_scenario(
                    quantile=q,
                    scenario_name=f"EVT P{q*100:.2f}% (1-em-{1/q:.0f})",
                )
                results.append(r)

        # Regime
        if self.regime_labels is not None:
            logger.info("Executando cenários de regime...")
            r = self.run_regime_stress(regime=1, quantile=0.05)
            if r is not None:
                results.append(r)

        self._results = results

        # Montar relatório
        report = self._build_report(results)

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"STRESS TEST COMPLETO: {len(results)} cenários")
            logger.info(f"{'='*60}")
            worst = report.nlargest(3, "portfolio_loss_pct")
            for _, row in worst.iterrows():
                logger.info(
                    f"  {row['scenario_name'][:40]:40s}  "
                    f"Perda: {row['portfolio_loss_pct']*100:.2f}%  "
                    f"CVaR: {row['cvar_1pct']*100:.2f}%"
                )

        return report

    def _build_report(self, results: List[StressResult]) -> pd.DataFrame:
        """Consolida resultados em DataFrame """
        rows = []
        for r in results:
            rows.append({
                "scenario_name":       r.scenario_name,
                "scenario_type":       r.scenario_type,
                "portfolio_return":    r.portfolio_return,
                "portfolio_loss_pct":  r.portfolio_loss_pct,
                "var_1pct":            r.var_1pct,
                "cvar_1pct":           r.cvar_1pct,
                "worst_asset":         r.worst_asset,
                "worst_asset_return":  r.worst_asset_return,
                "description":         r.description,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("portfolio_loss_pct", ascending=False).reset_index(drop=True)
        return df

    def get_results(self) -> List[StressResult]:
        """Retorna lista de StressResult dos últimos cenários executados"""
        return self._results

    def get_worst_scenario(self) -> Optional[StressResult]:
        """Retorna o cenário com maior perda"""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.portfolio_loss_pct)

    def save_report(
        self, report: pd.DataFrame, path: Union[str, Path]
    ):
        """Salva relatório em CSV"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(path, index=False)
        logger.info(f"Relatório de stress salvo em {path}")



def run_stress_tests(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    garch_results: Optional[Dict] = None,
    copula=None,
    regime_labels: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> Tuple[StressTester, pd.DataFrame]:
    """
    Wrapper para main_pipeline.py

    """
    tester = StressTester(
        returns_df=returns_df,
        weights=weights,
        garch_results=garch_results,
        copula=copula,
        regime_labels=regime_labels,
    )

    report = tester.run_all()

    if save_path:
        tester.save_report(report, save_path)

    return tester, report


# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    T = 1500
    n = 5
    tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "ABEV3.SA"]
    dates = pd.date_range("2019-01-01", periods=T, freq="B")

    # Simular retornos
    R_mat = 0.4 * np.ones((n, n)) + 0.6 * np.eye(n)
    L = np.linalg.cholesky(R_mat)
    returns_sim = (np.random.standard_normal((T, n)) @ L.T) * 0.015
    returns_df = pd.DataFrame(returns_sim, index=dates, columns=tickers)

    # Pesos
    weights = np.array([0.25, 0.20, 0.25, 0.15, 0.15])

    # Regime labels (simulado)
    regime_sim = (returns_df.mean(axis=1).rolling(21).std() > 0.012).astype(int)
    regime_sim = regime_sim.fillna(0)

    #  Run completo 
    tester, report = run_stress_tests(
        returns_df=returns_df,
        weights=weights,
        regime_labels=regime_sim,
    )

    print(f"\nCenários executados: {len(report)}")
    print(f"\nTop 5 piores cenários:")
    cols = ["scenario_name", "scenario_type", "portfolio_loss_pct", "cvar_1pct"]
    print(report[cols].head(5).round(4).to_string())

    print(f"\nPior cenário:")
    worst = tester.get_worst_scenario()
    if worst:
        print(worst.summary())

    print("\n Teste concluído.")
