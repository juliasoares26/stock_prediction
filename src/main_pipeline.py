import sys
import io
import csv
import time
import cProfile
import pstats
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from joblib import Parallel, delayed, Memory

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

#  Módulos de dados 
from data.data_loader import DataLoader
from data.preprocessor import preprocess_stocks_separately

#  Módulos marginais 
from marginals.semi_parametric import SemiParametricGARCH_EVT
from marginals.garch import GARCHFitter, fit_garch_all          
from marginals.evt_gev import GEVFitter, fit_gev_all           

#  Módulos de cópulas 
from copulas.vine_copulas import CVineCopula, DVineCopula
from copulas.estimation import PITTransformer                    
from copulas.selection import CopulaSelector                    

#  Módulos de risco (existentes) 
from risk.copula_var_es import CopulaEVTRisk, OptimizedCopulaRisk
from risk.tail_dependence import TailDependence

#  Módulos de risco (novos) 
from risk.return_levels import ReturnLevelCalculator            
from risk.stress_testing import StressTester, run_stress_tests  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / '.cache'
memory = Memory(location=CACHE_DIR, verbose=0)



def fit_single_marginal(asset: str, values: np.ndarray, index) -> tuple:
    """
    Ajusta modelo marginal GARCH-EVT para um ativo
    """
    try:
        series = pd.Series(values, index=index, name=asset)
        model = SemiParametricGARCH_EVT()
        model.fit(series, garch_spec='GARCH', garch_p=1, garch_q=1, dist='skewt')
        summary = model.get_summary()
        return asset, model, summary, None
    except Exception as e:
        return asset, None, None, str(e)


@memory.cache
def fit_single_marginal_cached(
    asset: str, values_bytes: bytes, shape: tuple, index_bytes: bytes
):
    values = np.frombuffer(values_bytes, dtype=np.float64).reshape(shape)
    index  = pd.to_datetime(np.frombuffer(index_bytes, dtype='int64'), unit='ns')
    return fit_single_marginal(asset, values, index)



# Utilitários de profiling


@contextmanager
def timed(label: str, timings: list):
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000
    timings.append({"label": label, "ms": round(elapsed_ms, 2)})
    logger.info(f" [{label}] {elapsed_ms:>10.1f} ms")


def _print_timings_table(timings: list):
    total = sum(t["ms"] for t in timings)
    logger.info("\n   TIMINGS ")
    for t in timings:
        pct = t["ms"] / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        logger.info(f"  {t['label']:<40} {t['ms']:>8.1f} ms  {pct:>5.1f}%  {bar}")
    logger.info(f"  {'TOTAL':<40} {total:>8.1f} ms")


def _save_profile_artifacts(pr: cProfile.Profile, timings: list):
    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumulative").print_stats(40)
    (RESULTS_DIR / "profile_step6.txt").write_text(stream.getvalue(), encoding="utf-8")
    pr.dump_stats(RESULTS_DIR / "profile_step6.prof")
    with open(RESULTS_DIR / "timings_step6.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "ms"])
        writer.writeheader()
        writer.writerows(timings)
    logger.info("  ✓ Artefatos de profiling salvos em results/")

# Pipeline 

class OptimizedPipeline:

    def __init__(
        self,
        use_cache: bool  = False,
        profile_step6: bool  = False,
        max_trees: int = 3,
        min_tau: float = 0.05,
        garch_model: str = "gjr",      
        pit_method: str = "empirical", 
        run_stress: bool = True,       
        run_return_levels: bool = True,     
        return_periods: list = None,        
    ):

        self.returns: pd.DataFrame = None
        self.asset_names: list = []
        self.marginal_models: dict = {}   # SemiParametricGARCH_EVT
        self.garch_results: dict = {}   # GARCHResult 
        self.std_residuals_df: pd.DataFrame = None 
        self.conditional_vol_df: pd.DataFrame = None 
        self.gev_fitter: GEVFitter = None 
        self.copula_model = None
        self.regime_labels: pd.Series = None
        self.results: dict = {}
        self._simulated_scenarios: np.ndarray = None

        self.use_cache = use_cache
        self.profile_step6 = profile_step6
        self.max_trees = max_trees
        self.min_tau = min_tau
        self.garch_model = garch_model
        self.pit_method = pit_method
        self.run_stress = run_stress
        self.run_return_levels = run_return_levels
        self.return_periods = return_periods or [1, 5, 10, 20, 50]

        logger.info(
            f"  max_trees={max_trees}  min_tau={min_tau}  "
            f"garch={garch_model}  pit={pit_method}  "
            f"stress={run_stress}  return_levels={run_return_levels}"
        )
        logger.info("=" * 80)

    # Carga e pré-processamento

    def step1_load_data(self, start_date: str = '2020-01-01') -> pd.DataFrame:
        logger.info("\n 1/8 Carregando dados")

        loader = DataLoader()
        loader.load_data()
        prices = loader.get_adjusted_close()

        logger.info(f"  Dados brutos: {prices.shape}")
        logger.info(f"  Período: {prices.index[0]} → {prices.index[-1]}")

        results = preprocess_stocks_separately(
            prices,
            return_type='log',
            handle_missing='forward_fill',
            winsorize=True,
            save=False,
        )

        self.returns     = results['stocks_returns']
        self.asset_names = list(self.returns.columns)

        logger.info(f"{len(self.asset_names)} ativos  {len(self.returns)} observações")
        return self.returns

    # Marginais semi-paramétricas (GARCH-EVT) —

    def step2a_fit_marginals_parallel(self) -> dict:
        logger.info("\n 2a/8 ajustando marginai semi-paramétricas")

        if self.use_cache:
            raw_results = [
                fit_single_marginal_cached(
                    asset,
                    self.returns[asset].values.tobytes(),
                    self.returns[asset].values.shape,
                    self.returns.index.view('int64').tobytes(),
                )
                for asset in self.asset_names
            ]
        else:
            raw_results = Parallel(n_jobs=-1, backend='loky', verbose=5)(
                delayed(fit_single_marginal)(
                    asset,
                    self.returns[asset].values,
                    self.returns.index,
                )
                for asset in self.asset_names
            )

        n_success = 0
        for asset, model, summary, error in raw_results:
            if model is not None:
                self.marginal_models[asset] = model
                n_success += 1
                logger.info(f"{asset}: AIC={summary['garch']['aic']:.2f}")
            else:
                logger.error(f"{asset}: {error}")

        logger.info(f"\n Marginais ajustadas: {n_success}/{len(self.asset_names)}")
        return self.marginal_models


    # ETAPA 2b  GARCH 

    def step2b_fit_garch(self) -> dict:
        """
        Ajusta GARCH explícito via garch.py (GARCHFitter)

       2a alimenta a vine, 2b alimenta EVT/stress
        """
        logger.info(f"\n 2b/8 ({self.garch_model.upper()})")

        std_resid, cond_vol, garch_results = fit_garch_all(
            returns_df=self.returns,
            model_type=self.garch_model,
            dist='normal',
            save_path=str(RESULTS_DIR / 'garch'),
        )

        self.std_residuals_df   = std_resid
        self.conditional_vol_df = cond_vol
        self.garch_results      = garch_results

        # Log persistência
        fitter = GARCHFitter(model_type=self.garch_model)
        fitter.results_ = garch_results
        fitter._is_fitted = True
        persist = fitter.get_persistence()
        logger.info(f"  Persistência média: {persist.mean():.4f}")
        logger.info(f"  Persistência max  : {persist.max():.4f} ({persist.idxmax()})")

        if (persist > 0.999).any():
            logger.warning(
                f"  ⚠ Ativos com persistência ≈ 1 (não-estacionário): "
                f"{list(persist[persist > 0.999].index)}"
            )

        logger.info(f"✓ GARCH: {len(garch_results)} ativos  "
                    f"resíduos {std_resid.shape}  vol {cond_vol.shape}")
        return garch_results

    # Transformação para uniforme 

    def step3_transform_to_uniform(self) -> np.ndarray:
        """
        Usa PITTransformer(method='empirical') de estimation.py
        """
        logger.info(f"\n 3/8 transformando para uniforme (PIT={self.pit_method})")

        n_obs    = len(self.returns)
        n_assets = len(self.asset_names)
        pseudo_obs = np.empty((n_obs, n_assets), dtype=np.float64)

        n_semiparametric = 0
        n_fallback = 0

        for i, asset in enumerate(self.asset_names):
            model = self.marginal_models.get(asset)

            # PIT semi-paramétrico de 2a
            if model is not None:
                try:
                    pseudo_obs[:, i] = model.probability_integral_transform()
                    n_semiparametric += 1
                    continue
                except Exception as e:
                    logger.debug(f"  {asset}: PIT semi-param falhou ({e}), usando fallback")

            # PITTransformer empírico (estimation.py)
            pit = PITTransformer(method=self.pit_method)
            series = self.returns[[asset]]
            u = pit.fit_transform(series)
            pseudo_obs[:, i] = u[:, 0]
            n_fallback += 1

        pseudo_obs = np.clip(pseudo_obs, 1e-6, 1 - 1e-6)

        logger.info(f"Semi-paramétrico: {n_semiparametric} ativos")
        logger.info(f"Fallback empírico: {n_fallback} ativos")
        logger.info(f"Pseudo-obs: {pseudo_obs.shape}"
                    f"range [{pseudo_obs.min():.4f}, {pseudo_obs.max():.4f}]")
        return pseudo_obs

    # Vine Copula

    def step4_fit_copula(
        self,
        pseudo_obs: np.ndarray,
        copula_type: str = 'cvine',
    ):
        """
        max_trees trunca vine após N árvores.
        min_tau ignora pares com dependência fraca.
        CopulaSelector de selection.py pode ser usado para pré-selecionar a melhor família por par antes da vine, passando o resultado como pair_families para CVineCopula.
        """
        logger.info(f"\n 4/8 Ajustando Cópula ({copula_type.upper()})...")
        logger.info(f"max_trees={self.max_trees}  min_tau={self.min_tau}")

        # pré-seleção de famílias por par 
        # Se CopulaSelector for suportado pela vine, passa pair_families
        pair_families = None
        try:
            if len(self.asset_names) <= 10:
                # Para d pequeno: seleção completa
                selector = CopulaSelector(
                    criterion='aic',
                    parallel=True,
                    tau_threshold=self.min_tau,
                )
                pair_results = selector.select_all_pairs(
                    pseudo_obs,
                    column_names=self.asset_names,
                    verbose=False,
                )
                # Montar dict {(i,j): family_name} para passar à vine
                pair_families = {
                    k: v.best_family
                    for k, v in pair_results.items()
                }
                dist = selector.get_family_distribution(pair_results)
                logger.info(f"  Famílias selecionadas: {dist.to_dict()}")
                self.results['copula_selection'] = selector.summarize_selection(pair_results)
            else:
                logger.info(
                    f"  d={len(self.asset_names)} > 10: "
                    f"pulando pré-seleção (use auto_select na vine)"
                )
        except Exception as e:
            logger.warning(f"Pré-seleção falhou ({e}). Vine usará auto_select interno.")

        fit_kwargs = dict(
            families=['gaussian', 'clayton', 'gumbel', 't'],
            auto_select=True,
        )
        if self.max_trees is not None:
            fit_kwargs['max_trees'] = self.max_trees
        if self.min_tau is not None:
            fit_kwargs['min_tau'] = self.min_tau
        if pair_families:
            fit_kwargs['pair_families'] = pair_families

        if copula_type == 'cvine':
            self.copula_model = CVineCopula(n_dim=len(self.asset_names))
            self.copula_model.fit(pseudo_obs, **fit_kwargs)
            logger.info(f"C-Vine: {len(self.copula_model.copulas)} pares")

        elif copula_type == 'dvine':
            fit_kwargs.pop('auto_select', None)
            self.copula_model = DVineCopula(n_dim=len(self.asset_names))
            self.copula_model.fit(pseudo_obs, **fit_kwargs)
            logger.info(f"D-Vine: {len(self.copula_model.copulas)} pares")

        return self.copula_model

    # Tail Dependence

    def step5_tail_dependence(self):
        logger.info("\n 5/8 Tail Dependence")

        td = TailDependence()
        td.fit(self.returns, method='empirical')

        lambda_L = td.get_lambda_L_dataframe()
        lambda_U = td.get_lambda_U_dataframe()

        logger.info(f"λ_L médio: {td._mean_off_diagonal(td.lambda_L_matrix):.4f}")
        logger.info(f"λ_U médio: {td._mean_off_diagonal(td.lambda_U_matrix):.4f}")

        lambda_L.to_csv(RESULTS_DIR / 'tail_dependence_lower.csv')
        lambda_U.to_csv(RESULTS_DIR / 'tail_dependence_upper.csv')

        self.results['tail_dependence'] = {'lambda_L': lambda_L, 'lambda_U': lambda_U}
        return td

    # 6a  VaR/ES via cópula 

    def _run_step6a_logic(
        self,
        weights: np.ndarray,
        n_simulations: int,
        seed: int,
        timings: list,
    ) -> tuple:

        with timed("copula_risk.fit", timings):
            copula_risk = CopulaEVTRisk()
            copula_risk.fit(
                returns=self.returns,
                marginal_models=self.marginal_models,
                copula_model=self.copula_model,
                asset_names=self.asset_names,
            )
            opt_risk = OptimizedCopulaRisk(copula_risk)

        with timed("copula_model.simulate", timings):
            try:
                self._simulated_scenarios = self.copula_model.simulate(
                    n_simulations, seed=seed
                )
                use_shared = True
                logger.info(f" shape={self._simulated_scenarios.shape}")
            except (AttributeError, NotImplementedError):
                self._simulated_scenarios = None
                use_shared = False
                logger.warning("simulate() não disponível — seed fixo por etapa")

        def _kw(extra: dict = None) -> dict:
            kw = dict(n_simulations=n_simulations, seed=seed)
            if extra:
                kw.update(extra)
            return kw

        with timed("portfolio_var_es_batch", timings):
            var_es_df = opt_risk.portfolio_var_es_batch(
                weights, confidence_levels=[0.95, 0.99, 0.995],
                n_simulations=n_simulations, seed=seed
            )

        with timed("component_var_parallel", timings):
            cvar_df = opt_risk.component_var_parallel(
                weights, confidence_level=0.99, **_kw()
            )

        with timed("compare_with_without_copula", timings):
            comparison = copula_risk.compare_with_without_copula(
                weights, confidence_level=0.99, **_kw()
            )

        return opt_risk, var_es_df, cvar_df, comparison

    def step6a_portfolio_risk(
        self,
        weights: np.ndarray = None,
        n_simulations: int = 10_000,
        seed: int = 42,
    ):
        logger.info("\n 6a/8 VaR/ES")

        if weights is None:
            weights = np.ones(len(self.asset_names)) / len(self.asset_names)
        self._weights = weights

        timings: list = []

        if self.profile_step6:
            pr = cProfile.Profile()
            pr.enable()
            opt_risk, var_es_df, cvar_df, comparison = self._run_step6a_logic(
                weights, n_simulations, seed, timings
            )
            pr.disable()
            _print_timings_table(timings)
            _save_profile_artifacts(pr, timings)
        else:
            opt_risk, var_es_df, cvar_df, comparison = self._run_step6a_logic(
                weights, n_simulations, seed, timings
            )
            _print_timings_table(timings)

        logger.info(f"\n{var_es_df}")
        var_es_df.to_csv(RESULTS_DIR / 'portfolio_var_es.csv', index=False)
        cvar_df.to_csv(RESULTS_DIR / 'component_var.csv', index=False)
        comparison.to_csv(RESULTS_DIR / 'copula_vs_independence.csv', index=False)

        self.results['var_es']        = var_es_df
        self.results['component_var'] = cvar_df
        self.results['comparison']    = comparison

        return opt_risk

    #  6b — Return Levels GEV 

    def step6b_return_levels(self, block_size: int = 22) -> pd.DataFrame:
        """
        Calcula return levels com IC bootstrap para cada ativo e para o portfólio usando evt_gev.py E return_levels.py.

        Usa std_residuals_df (saída do step2b / GARCH explícito) se disponível, pois a GEV sobre resíduos padronizados é mais estável do que sobre retornos brutos
        Fallback: retornos bruto
        """
        if not self.run_return_levels:
            logger.info("\n Return Levels (run_return_levels=False)")
            return pd.DataFrame()

        logger.info(f"\n Return Levels GEV (block_size={block_size} dias)")

        # Escolher dados: resíduos padronizados são preferíveis
        if self.std_residuals_df is not None and not self.std_residuals_df.empty:
            data_for_gev = self.std_residuals_df
            logger.info("Usando resíduos padronizados GARCH")
        else:
            data_for_gev = self.returns
            logger.info("std_residuals não disponível, usando retornos brutos")

        # Ajustar GEV para cada ativo
        self.gev_fitter = GEVFitter()
        self.gev_fitter.fit_all(data_for_gev, block_size=block_size)

        # Calcular return levels com IC bootstrap
        rlc = ReturnLevelCalculator(
            ci_method='bootstrap',
            n_bootstrap=300,
            portfolio_method='monte_carlo',
        )

        weights = getattr(self, '_weights', None)
        if weights is None:
            weights = np.ones(len(self.asset_names)) / len(self.asset_names)

        rl_df = rlc.compute_all(
            gev_fitter=self.gev_fitter,
            returns_df=self.returns,
            weights=weights,
            return_periods_years=self.return_periods,
            block_size=block_size,
            copula=self.copula_model,
        )

        # Salvar
        rl_df.to_csv(RESULTS_DIR / 'return_levels.csv', index=False)
        self.gev_fitter.get_params_df().to_csv(RESULTS_DIR / 'gev_params.csv')

        # Log resumo do portfólio
        port_df = rl_df[rl_df['ticker'] == 'PORTFOLIO'][
            ['years', 'estimate', 'ci_lower', 'ci_upper']
        ]
        if not port_df.empty:
            logger.info("\n  Return Levels do Portfólio:")
            for _, row in port_df.iterrows():
                logger.info(
                    f"    {row['years']:.0f}Y: "
                    f"{row['estimate']*100:+.2f}% "
                    f"IC [{row['ci_lower']*100:+.2f}%, {row['ci_upper']*100:+.2f}%]"
                )

        self.results['return_levels'] = rl_df
        logger.info(f" Return levels: {len(rl_df)} entradas  "
                    f"salvos em results/return_levels.csv")
        return rl_df

    # 6c — Stress Testing 

    def step6c_stress_testing(self) -> pd.DataFrame:
        if not self.run_stress:
            logger.info("\n Stress Testing (run_stress=False)")
            return pd.DataFrame()

        logger.info("\n Stress Testing")

        weights = getattr(self, '_weights', None)
        if weights is None:
            weights = np.ones(len(self.asset_names)) / len(self.asset_names)

        tester, stress_report = run_stress_tests(
            returns_df=self.returns,
            weights=weights,
            garch_results=self.garch_results, # de step2b
            copula=self.copula_model, # de step4
            regime_labels=self.regime_labels, # None se não calculado
            save_path=str(RESULTS_DIR / 'stress_report.csv'),
        )

        # Log top 5 piores
        logger.info(f"\n  Top 5 piores cenários:")
        cols = ['scenario_name', 'scenario_type', 'portfolio_loss_pct', 'cvar_1pct']
        for _, row in stress_report[cols].head(5).iterrows():
            logger.info(
                f"    {row['scenario_name'][:35]:35s}  "
                f"tipo={row['scenario_type']:12s}  "
                f"perda={row['portfolio_loss_pct']*100:.2f}%  "
                f"CVaR={row['cvar_1pct']*100:.2f}%"
            )

        # Pior cenário absoluto
        worst = tester.get_worst_scenario()
        if worst:
            logger.info(f"\n  Pior cenário absoluto: {worst.scenario_name}")
            logger.info(f"  Perda máxima: {worst.portfolio_loss_pct*100:.2f}%")

        self.results['stress_report'] = stress_report
        logger.info(f"{len(stress_report)} cenários  salvos em results/stress_report.csv")
        return stress_report

    # Relatório 

    def step7_generate_report(self) -> Path:
        logger.info("\n Relatório")

        report_path = RESULTS_DIR / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Período: {self.returns.index[0]} → {self.returns.index[-1]}\n")
            f.write(f"Ativos: {len(self.asset_names)}\n")
            f.write(f"Observações: {len(self.returns)}\n")
            f.write(f"GARCH: {self.garch_model}\n")
            f.write(f"PIT: {self.pit_method}\n")
            f.write(f"max_trees: {self.max_trees}\n")
            f.write(f"min_tau: {self.min_tau}\n\n")

            # VaR/ES
            for title, key in [
                ("VaR e ES do portfolio",   'var_es'),
                ("Component VaR (top 10)",  'component_var'),
                ("Copula vs independencia", 'comparison'),
            ]:
                if key in self.results:
                    df = self.results[key]
                    if key == 'component_var':
                        df = df.nlargest(10, 'Contribution_%')
                    f.write(df.to_string(index=False))
                f.write("\n\n")

            # Return Levels
            if 'return_levels' in self.results:
                port_rl = self.results['return_levels']
                port_rl = port_rl[port_rl['ticker'] == 'PORTFOLIO']
                if not port_rl.empty:
                    f.write(
                        port_rl[['years', 'estimate', 'ci_lower', 'ci_upper', 'method']]
                        .round(4).to_string(index=False)
                    )
            f.write("\n\n")

            # Stress Testing
            if 'stress_report' in self.results:
                sr = self.results['stress_report']
                cols = ['scenario_name', 'scenario_type', 'portfolio_loss_pct', 'cvar_1pct']
                f.write(sr[cols].head(10).round(4).to_string(index=False))
    
        logger.info(f"Relatório salvo em: {report_path}")
        return report_path

    # Pipeline completo 

    def run_full_pipeline(
        self,
        copula_type: str = 'cvine',
        weights: np.ndarray = None,
        n_simulations: int = 10_000,
        seed: int = 42,
        block_size: int = 22,
    ) -> dict:
        try:
            self.step1_load_data()
            self.step2a_fit_marginals_parallel()
            self.step2b_fit_garch()                             
            pseudo_obs = self.step3_transform_to_uniform()       
            self.step4_fit_copula(pseudo_obs, copula_type)       
            self.step5_tail_dependence()
            self.step6a_portfolio_risk(
                weights=weights,
                n_simulations=n_simulations,
                seed=seed,
            )
            self.step6b_return_levels(block_size=block_size)    
            self.step6c_stress_testing()                         
            self.step7_generate_report()

           
            logger.info("Pipeline Concluída!")
            logger.info(f"  Resultados em: {RESULTS_DIR}")

            return self.results

        except Exception as e:
            logger.error(f"Erro no pipeline: {str(e)}", exc_info=True)
            raise

    def save_state(self, path: Path = None):
        """Salva estado completo pós-etapas 1–5 para reuso"""
        import joblib
        path = path or RESULTS_DIR / 'pipeline_state.pkl'
        joblib.dump({
            'returns': self.returns,
            'marginal_models': self.marginal_models,
            'garch_results': self.garch_results,       
            'std_residuals_df': self.std_residuals_df,   
            'conditional_vol_df': self.conditional_vol_df,  
            'gev_fitter': self.gev_fitter,         
            'copula_model': self.copula_model,
            'asset_names': self.asset_names,
            'regime_labels': self.regime_labels,
        }, path)
        logger.info(f"Estado salvo em: {path}")

    @classmethod
    def load_state(cls, path: Path, **pipeline_kwargs) -> 'OptimizedPipeline':
        """Carrega estado salvo e retorna pipeline pronto para etapas após 6."""
        import joblib
        state = joblib.load(path)
        pipeline = cls(**pipeline_kwargs)
        for k, v in state.items():
            setattr(pipeline, k, v)
        logger.info(f"Estado carregado de: {path}")
        return pipeline


# Entry point

if __name__ == "__main__":
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description="Optimized Copula e EVT Pipeline")
    parser.add_argument("--copula",        default="cvine",     choices=["cvine", "dvine"])
    parser.add_argument("--n-sim",         default=10_000,      type=int)
    parser.add_argument("--seed",          default=42,          type=int)
    parser.add_argument("--max-trees",     default=3,           type=int)
    parser.add_argument("--min-tau",       default=0.05,        type=float)
    parser.add_argument("--garch-model",   default="gjr",       choices=["garch", "gjr", "auto"])
    parser.add_argument("--pit-method",    default="empirical", choices=["empirical", "normal"])
    parser.add_argument("--block-size",    default=22,          type=int,
                        help="Tamanho do bloco GEV em dias (22≈mensal, 63≈trimestral)")
    parser.add_argument("--no-stress",     action="store_true",
                        help="Desativa stress testing")
    parser.add_argument("--no-rl",         action="store_true",
                        help="Desativa return levels")
    parser.add_argument("--cache",         action="store_true")
    parser.add_argument("--profile",       action="store_true")
    parser.add_argument("--save-state",    action="store_true")
    args = parser.parse_args()

   
    t0 = time.time()

    pipeline = OptimizedPipeline(
        use_cache=args.cache,
        profile_step6=args.profile,
        max_trees=args.max_trees,
        min_tau=args.min_tau,
        garch_model=args.garch_model,
        pit_method=args.pit_method,
        run_stress=not args.no_stress,
        run_return_levels=not args.no_rl,
    )

    results = pipeline.run_full_pipeline(
        copula_type=args.copula,
        n_simulations=args.n_sim,
        seed=args.seed,
        block_size=args.block_size,
    )

    if args.save_state:
        pipeline.save_state()

    elapsed = time.time() - t0
    logger.info(f"\n Concluído em {elapsed:.1f}s  Resultados em: {RESULTS_DIR}")

