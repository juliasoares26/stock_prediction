import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numba
from numba import jit, prange

@jit(nopython=True, cache=True, fastmath=True)
def fast_portfolio_returns(simulated_returns, weights):
    return simulated_returns @ weights

@jit(nopython=True, cache=True, parallel=True)
def fast_quantile_batch(data, quantiles):
    n = len(quantiles)
    result = np.empty(n)
    for i in prange(n):
        result[i] = np.quantile(data, quantiles[i])
    return result

@jit(nopython=True, cache=True)
def fast_es_calculation(returns, var_threshold):
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) > 0:
        return tail_returns.mean()
    return var_threshold

class CopulaEVTRisk:
  def __init__(self):
    self.marginal_models = {} # garch-evt por ativo
    self.copula_model = None # Vine ou DCC
    self.asset_names = None
    self.fitted = False
    self.marginal_cdfs = {} # cdfs semi-parametricas pre-computadas
    self.marginal_inv_cdfs = {}

  def fit(
      self,
      returns: pd.DataFrame,
      marginal_models: Dict,
      copula_model,
      asset_names: Optional[List[str]] = None
  ):
    self.marginal_models = marginal_models
    self.copula_model = copula_model

    if asset_names is None:
      self.asset_names = list(returns.columns)
    else:
      self.asset_names = asset_names

    self._precompute_marginal_transforms()

    self.fitted = True

    logger.info(f"CopulaEVTRisk configurado: {len(self.asset_names)} ativos")

  def _precompute_marginal_transforms(self):
    logger.info("pré-computando transformações marginais EVT")

    for asset_name, model in self.marginal_models.items():
      returns_grid = np.linspace(
        model.returns.min() - 4*model.returns.std(),
        model.returns.max() + 4*model.returns.std(),
        2000
      )

      cdf_values = np.vstack([
        self._semi_parametric_cdf(r, model)
        for r in returns_grid
      ])

      cdf_values = cdf_values.flatten() 
      unique_mask = np.concatenate([[True], np.diff(cdf_values) > 1e-10])
      returns_grid_unique = returns_grid[unique_mask]
      cdf_values_unique = cdf_values[unique_mask]

      self.marginal_cdfs[asset_name] = interp1d(
        returns_grid_unique,
        cdf_values_unique,
        kind='linear',
        bounds_error=False,
        fill_value=(0.0, 1.0)
      )

      self.marginal_inv_cdfs[asset_name] = interp1d(
        cdf_values_unique,
        returns_grid_unique,
        kind ='linear',
        bounds_error=False,
        fill_value=(returns_grid_unique[0], returns_grid_unique[-1])
      )

    logger.info("transformações EVT pré-computadas")

  # cdf semiparametrica, evt empirica no centro e gpd nas caudas
  def _semi_parametric_cdf(
      self,
      x:float,
      model,
      lower_quantile: float = 0.05,
      upper_quantile: float = 0.95
  ) -> float:
    
    threshold_lower = np.quantile(model.returns, lower_quantile)
    threshold_upper = np.quantile(model.returns, upper_quantile)

    if x < threshold_lower:
      if hasattr(model, 'gpd_left') and model.gpd_left is  not None:
        excess = threshold_lower - x
         # P(X <= x) = P(X <= threshold_lower) * P(X <= x | X <= threshold_lower)
         # Para GPD: P(excess > e) = (1 + xi*e/beta)^(-1/xi)
        cdf_excess = model.gpd_left.cdf(excess)
        return lower_quantile * (1 - cdf_excess)
      else:
        return np.mean(model.returns <= x)
      
    elif x > threshold_upper:
      if hasattr(model, 'gpd_right') and model.gpd_right is not None:
        excess = x - threshold_upper
        cdf_excess = model.gpd_right.cdf(excess)
        return upper_quantile + (1 - upper_quantile) * cdf_excess
      else:
        return np.mean(model.returns <= x)
      
    else:
      # ajusta para continuidade nos thresholds
      emp_cdf = np.mean(model.returns <= x)

      # normaliza para [lower_quantile, upper_quantile]
      emp_cdf_center = np.mean((model.returns >= threshold_lower) & (model.returns <= x))
      total_center = np.mean((model.returns >= threshold_lower) & (model.returns <= threshold_upper))
      
      if total_center > 0:
        return lower_quantile + (upper_quantile - lower_quantile) * (emp_cdf_center / total_center)
      else:
        return emp_cdf
  
  # transforma retorno em pseudo-observações [0,1] usando cdfs semi-parametricas
  def transform_to_uniform(self, returns: pd.DataFrame) -> np.ndarray:
    n_obs = len(returns)
    n_assets = len(self.asset_names)
    
    uniform_data = np.zeros((n_obs, n_assets))

    for i, asset in enumerate(self.asset_names):
      if asset not in self.marginal_cdfs:
        logger.warning(f"cdf não encontrada para {asset}, usando rankdata")
        uniform_data[:, i] = stats.rankdata(returns[asset]) / (len(returns) + 1)
      else:
        cdf_func = self.marginal_cdfs[asset]
        asset_returns = returns[asset].values
        uniform_data[:, i] = cdf_func(asset_returns)

        uniform_data[:, i] = np.clip(uniform_data[:, i], 1e-10, 1 - 1e-10)

    logger.info(f"transformando para uniformes via CDF EVT: {uniform_data.shape}")

    return uniform_data
  
  def simulate_portfolio_returns(
      self, 
      weights: np.ndarray,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None
  ) -> np.ndarray:
    
    if not self.fitted:
      raise ValueError("Execute fit() primeiro")
    
    if seed is not None:
      np.random.seed(seed)

    logger.info(f"simulado {n_simulations} cenários (h={horizon})")

    # pseudo observações uniformes correlacionadas via cópulas

    if use_copula:
      if self.copula_model is None:
        raise ValueError("copula não definida")
      
      # usando cópia para capturar dependencia
      if hasattr(self.copula_model, 'simulate'):
        #cvinecopula, dccopula ou similar
        uniform_samples = self.copula_model.simulate(
          n_samples=n_simulations,
          seed=seed
        )
      else:
        if hasattr(self.copula_model, 'rvs'):
          uniform_samples = self.copula_model.rvs(n_simulations)
        else:
          raise ValueError("copula não tem método simulate() ou rvs()")
        
    else:
     uniform_samples = np.random.uniform(
       1e-10, 1 - 1e-10,
       (n_simulations, len(self.asset_names))
     )

    simulated_returns = np.zeros((n_simulations, len(self.asset_names)))

    for i, asset in enumerate(self.asset_names):
      u_samples = uniform_samples[:, i]

      u_samples = np.clip(u_samples, 1e-10, 1 - 1e-10)
      if asset in self.marginal_inv_cdfs:
        inv_cdf_func = self.marginal_inv_cdfs[asset]
        simulated_returns[:, i] = inv_cdf_func(u_samples)
      else:
        logger.warning(f"inverse CDF não encontrada para {asset}")
        model = self.marginal_models[asset]
        simulated_returns[:, i] = np.quantile(
          model.returns,
          u_samples
        )
    # ajusta para horizonte multi-periodo
    if horizon > 1:
      #escala por sqrt 
      simulated_returns *= np.sqrt(horizon)

      # retornos do portfolio
    portfolio_returns = simulated_returns @ weights

    logger.info(f"simulação concluída (copula={use_copula})")
    logger.info(f"retornosμ={portfolio_returns.mean():.4f}, σ={portfolio_returns.std():.4f}")

    return portfolio_returns

  def portfolio_var(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.95,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None
  ) -> float:
    
    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula, seed
    )

    var = np.quantile(portfolio_returns, 1 - confidence_level)

    logger.info(f"VaR {confidence_level:.0%}: {var:.4f}")

    return var
  
  def portfolio_es(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.95,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None
  ) -> float:
    
    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula, seed
    )

    var = np.quantile(portfolio_returns, 1 - confidence_level)
    es = portfolio_returns[portfolio_returns <= var].mean()

    logger.info(f"ES {confidence_level:.0%}: {es:.4f}")
    logger.info(f"ES/VAR ratio: {es/var:.3f}")

    return es
  
  def tail_var_es(
      self,
      weights: np.ndarray,
      tail_quantiles: List[float] = [0.95, 0.99, 0.995, 0.999],
      n_simulations: int = 50000,
      horizon: int = 1,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    
    # var e es para multiplos quantis de cauda

    logger.info(f"calculando tail VaR/ES para {len(tail_quantiles)} quantis")

    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula=True, seed=seed
    )

    results = []

    for q in tail_quantiles:
      var_q = np.quantile(portfolio_returns, 1 - q)
      es_q = portfolio_returns[portfolio_returns <= var_q].mean()

      # num de obs na cauda
      n_tail = np.sum(portfolio_returns <= var_q)

      results.append({
        'Confidence_Level': q,
        'VaR': var_q,
        'ES': es_q,
        'ES_VaR_Ratio': es_q / var_q if var_q != 0 else np.nan,
        'N_Tail_Obs': n_tail,
        'Tail_Pct': 100 * n_tail / len(portfolio_returns)
      })

    df = pd.DataFrame(results)
    logger.info(f"Tail VaR/ES calculado")

    return df
  
  def compare_with_without_copula(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.99,
      n_simulations: int = 1000,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    
    # VaR/ES com e sem copula

    logger.info("comparando: copula vs independencia")

    # com copula
    var_copula = self.portfolio_var(weights, confidence_level, n_simulations, 1, True, seed)
    es_copula = self.portfolio_es(weights, confidence_level, n_simulations, 1, True, seed)

    # sem copula
    var_indep = self.portfolio_var(weights, confidence_level, n_simulations, 1, False, seed)
    es_indep = self.portfolio_es(weights, confidence_level, n_simulations, 1, False, seed)

    comparison = pd.DataFrame({
      'Method': ['Copula-EVT', 'Independence'],
      'VaR': [var_copula, var_indep],
      'ES': [es_copula, es_indep],
      'ES_VaR_Ratio': [es_copula/var_copula, es_indep/var_indep]
    })
  
    comparison['VaR_Diff_%'] = 100 * (comparison['VaR'] - var_indep) / abs(var_indep)
    comparison['ES_Diff_%'] = 100 * (comparison['ES'] - es_indep) / abs(es_indep)

    logger.info(f"var: {comparison.loc[0, 'VaR_Diff_%']:.2f}% maior")
    logger.info(f"ES: {comparison.loc[0, 'ES_Diff_%']:.2f}% maior")

    return comparison

  def scenario_analysis(
      self,
      weights: np.ndarray,
      scenarios: Dict[str, Dict],
      n_simulations: int = 1000,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    
    logger.info(f"Análise de {len(scenarios)} cenários")

    original_marginals = {}
    for asset, model in self.marginal_models.items():
      original_marginals[asset] = {
        'gpd_left_xi': model.gpd_left.c if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'c') else None,
        'gpd_right_xi': model.gpd_right.c if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'c') else None,
        'volatility': model.returns.std()
     }

    if hasattr(self.copula_model, 'corr') or hasattr(self.copula_model, 'tau'):
      original_corr = self._extract_copula_correlation()
    else:
      original_corr = None

    results = []

    for name, params in scenarios.items():
      logger.info(f"  Cenário: {name}")
            
      tail_mult = params.get('tail_multiplier', 1.0)
      corr_mult = params.get('correlation_multiplier', 1.0)
      vol_mult = params.get('volatility_multiplier', 1.0)
            
      # modifica marginais EVT
      self._apply_marginal_stress(tail_mult, vol_mult)
            
      # modifica copula (correlações/dependência)
      self._apply_copula_stress(corr_mult, original_corr)
            
      # Re-computa transformações com parâmetros stressed
      self._precompute_marginal_transforms()
            
      # calcula VaR/ES sob cenário estressado
      var_stressed = self.portfolio_var(
        weights, 0.99, n_simulations, use_copula=True, seed=seed
      )
      es_stressed = self.portfolio_es(
        weights, 0.99, n_simulations, use_copula=True, seed=seed
      )
            
      results.append({
      'Scenario': name,
      'Tail_Multiplier': tail_mult,
      'Corr_Multiplier': corr_mult,
      'Vol_Multiplier': vol_mult,
      'VaR_99': var_stressed,
      'ES_99': es_stressed,
      'ES_VaR_Ratio': es_stressed / var_stressed if var_stressed != 0 else np.nan
      })
            
      # restaura parâmetros originais
      self._restore_marginals(original_marginals)
      if original_corr is not None:
        self._restore_copula_correlation(original_corr)
        
      # Re-computa transformações com parâmetros originais
      self._precompute_marginal_transforms()
        
      df = pd.DataFrame(results)
      logger.info("Análise de cenários concluída")
        
      return df
    
  def _apply_marginal_stress(
        self,
        tail_multiplier: float,
        volatility_multiplier: float
    ):
        """
        Aplica stress aos parâmetros das marginais EVT
        
        tail_multiplier: Amplifica shape parameter ξ do GPD (caudas mais pesadas)
        volatility_multiplier: Amplifica volatilidade dos retornos
        """
        for asset, model in self.marginal_models.items():
            # Stress no shape parameter GPD (caudas)
            if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'c'):
                # c é o shape parameter ξ no scipy
                # ξ > 0: cauda mais pesada (distribuição de Pareto)
                model.gpd_left.c *= tail_multiplier
            
            if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'c'):
                model.gpd_right.c *= tail_multiplier
            
            # Stress na volatilidade (amplia dispersão)
            if volatility_multiplier != 1.0:
                mean_return = model.returns.mean()
                model.returns = mean_return + (model.returns - mean_return) * volatility_multiplier
    
  def _apply_copula_stress(
        self,
        correlation_multiplier: float,
        original_corr: Optional[np.ndarray]
    ):
        """
        Aplica stress à estrutura de dependência da copula
        
        correlation_multiplier: Amplifica correlações 
        """
        if correlation_multiplier == 1.0 or original_corr is None:
            return
        
        # Para Gaussian ou t-Student copula: amplia correlações
        if hasattr(self.copula_model, 'corr'):
            # Amplia correlações mantendo diagonal = 1
            stressed_corr = original_corr.copy()
            n = len(stressed_corr)
            
            for i in range(n):
                for j in range(i+1, n):
                    # Amplifica correlação mas mantém em [-1, 1]
                    stressed_corr[i, j] = np.clip(
                        original_corr[i, j] * correlation_multiplier,
                        -0.99, 0.99
                    )
                    stressed_corr[j, i] = stressed_corr[i, j]
            
            # Garante que matriz é positiva definida
            stressed_corr = self._nearest_positive_definite(stressed_corr)
            
            self.copula_model.corr = stressed_corr
        
        elif hasattr(self.copula_model, 'tau'):
            # Para Archimedean copulas: amplia Kendall's tau
            original_tau = self.copula_model.tau
            stressed_tau = np.clip(
                original_tau * correlation_multiplier,
                -0.99, 0.99
            )
            self.copula_model.tau = stressed_tau
    
  def _extract_copula_correlation(self) -> Optional[np.ndarray]:
        """Extrai matriz de correlação/dependência da copula"""
        if hasattr(self.copula_model, 'corr'):
            return self.copula_model.corr.copy()
        elif hasattr(self.copula_model, 'tau'):
            # Para Archimedean: retorna tau como proxy
            return self.copula_model.tau
        else:
            return None
    
  def _restore_marginals(self, original_params: Dict):
        """Restaura parâmetros originais das marginais"""
        for asset, params in original_params.items():
            model = self.marginal_models[asset]
            
            if params['gpd_left_xi'] is not None:
                if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'c'):
                    model.gpd_left.c = params['gpd_left_xi']
            
            if params['gpd_right_xi'] is not None:
                if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'c'):
                    model.gpd_right.c = params['gpd_right_xi']
    
  def _restore_copula_correlation(self, original_corr):
        """Restaura correlação original da copula"""
        if hasattr(self.copula_model, 'corr'):
            self.copula_model.corr = original_corr.copy()
        elif hasattr(self.copula_model, 'tau'):
            self.copula_model.tau = original_corr
    
  def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Encontra a matriz positiva definida mais próxima de A
        Método de Higham (1988)
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
  def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Verifica se matriz é positiva definida"""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
  def marginal_contribution_to_risk(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 1000,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Contribuição marginal de cada ativo para VaR/ES
        
        MCVaR_i = ∂VaR/∂w_i
        """
        logger.info("Calculando contribuições marginais...")
        
        # VaR base do portfólio
        var_base = self.portfolio_var(weights, confidence_level, n_simulations, seed=seed)
        es_base = self.portfolio_es(weights, confidence_level, n_simulations, seed=seed)
        
        epsilon = 0.001
        contributions = []
        
        for i, asset in enumerate(self.asset_names):
            # Perturba peso do ativo i
            w_perturbed = weights.copy()
            w_perturbed[i] += epsilon
            w_perturbed = w_perturbed / w_perturbed.sum()  # Renormaliza
            
            # Calcula VaR/ES com peso perturbado
            var_perturbed = self.portfolio_var(w_perturbed, confidence_level, n_simulations, seed=seed)
            es_perturbed = self.portfolio_es(w_perturbed, confidence_level, n_simulations, seed=seed)
            
            # Derivadas numéricas
            marginal_var = (var_perturbed - var_base) / epsilon
            marginal_es = (es_perturbed - es_base) / epsilon
            
            # Contribuições componentes
            component_var = weights[i] * marginal_var
            component_es = weights[i] * marginal_es
            
            contributions.append({
                'Asset': asset,
                'Weight': weights[i],
                'Marginal_VaR': marginal_var,
                'Component_VaR': component_var,
                'Contribution_VaR_%': 100 * component_var / var_base,
                'Marginal_ES': marginal_es,
                'Component_ES': component_es,
                'Contribution_ES_%': 100 * component_es / es_base
            })
        
        df = pd.DataFrame(contributions)
        logger.info("Contribuições calculadas")
        
        return df
    
  def backtesting_violations(
        self,
        weights: np.ndarray,
        realized_returns: pd.Series,
        confidence_level: float = 0.99,
        rolling_window: int = 252
    ) -> Dict:
        """
        Backtesting: conta violações do VaR
        
        Testes:
        Kupiec POF 
        Christoffersen - independencia
        
        Argumentos
        realized_returns: Retornos realizados do portfólio
        rolling_window: Janela para re-estimação
    
        """
        logger.info("Backtesting VaR...")
        
        violations = []
        var_forecasts = []
        
        # Implementação simplificada 
        var = self.portfolio_var(weights, confidence_level, n_simulations=1000)
        
        for t in range(len(realized_returns)):
            var_forecasts.append(var)
            violations.append(1 if realized_returns.iloc[t] < var else 0)
        
        violations = np.array(violations)
        n_violations = violations.sum()
        n_obs = len(violations)
        violation_rate = n_violations / n_obs if n_obs > 0 else 0
        expected_rate = 1 - confidence_level
        
        # Kupiec (Likelihood Ratio)
        if n_violations > 0 and violation_rate > 0:
            lr_stat = -2 * (
                n_violations * np.log(expected_rate) + 
                (n_obs - n_violations) * np.log(1 - expected_rate) -
                n_violations * np.log(violation_rate) -
                (n_obs - n_violations) * np.log(1 - violation_rate)
            )
        else:
            lr_stat = np.nan
        
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, df=1) if not np.isnan(lr_stat) else np.nan
        
        logger.info(f"Backtesting: {n_violations}/{n_obs} violações ({violation_rate:.2%})")
        logger.info(f"Esperado: {expected_rate:.2%}, p-value: {p_value:.4f}")
        
        return {
            'n_violations': n_violations,
            'n_observations': n_obs,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_lr': lr_stat,
            'kupiec_pvalue': p_value,
            'test_passed': p_value > 0.05 if not np.isnan(p_value) else False
        }

class OptimizedCopulaRisk:

    def __init__(self, copula_risk: CopulaEVTRisk):
        self.copula_risk = copula_risk
        self.n_jobs = -1  # Usa todos os cores
    
    def portfolio_var_es_batch(
        self,
        weights: np.ndarray,
        confidence_levels: list = [0.95, 0.99, 0.995],
        n_simulations: int = 10000,
        seed: int = 42
    ):
        """
        Calcula VaR e ES para múltiplos níveis de confiança
        """
        logger.info(f"Simulando {n_simulations} cenários (otimizado)...")
        
        # única simulação para todos os níveis
        np.random.seed(seed)
        portfolio_returns = self.copula_risk.simulate_portfolio_returns(
            weights, n_simulations, horizon=1, use_copula=True
        )
        
        # Calcula todos os quantis de uma vez com Numba
        quantiles = np.array([1 - cl for cl in confidence_levels])
        vars = fast_quantile_batch(portfolio_returns, quantiles)
        
        # Calcula ES para cada VaR
        results = []
        for i, cl in enumerate(confidence_levels):
            var = vars[i]
            es = fast_es_calculation(portfolio_returns, var)
            
            results.append({
                'Confidence': cl,
                'VaR': var,
                'ES': es,
                'ES_VaR_Ratio': es / var if var != 0 else np.nan
            })
        
        return pd.DataFrame(results)
    
    def component_var_parallel(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 1000,
        epsilon: float = 0.01,
        seed: int = 42
    ):
        """
        Calcula Component VaR em paralelo 
        """
        logger.info("Calculando Component VaR (paralelo)...")
        
        n_assets = len(weights)
        
        # VaR base
        var_base = self.copula_risk.portfolio_var(
            weights, confidence_level, n_simulations, seed=seed
        )
        
        # Função para calcular VaR de um ativo perturbado
        def calc_perturbed_var(i):
            w_perturbed = weights.copy()
            w_perturbed[i] += epsilon
            w_perturbed = w_perturbed / w_perturbed.sum()  # Renormaliza
            
            var_perturbed = self.copula_risk.portfolio_var(
                w_perturbed, confidence_level, n_simulations, seed=seed+i
            )
            
            marginal_var = (var_perturbed - var_base) / epsilon
            component_var = marginal_var * weights[i]
            
            return {
                'Asset': self.copula_risk.asset_names[i],
                'Weight': weights[i],
                'Marginal_VaR': marginal_var,
                'Component_VaR': component_var,
                'Contribution_%': 100 * component_var / abs(var_base)
            }
        
        # Executa em paralelo
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(calc_perturbed_var)(i) for i in range(n_assets)
        )
        
        return pd.DataFrame(results)



if __name__ == "__main__":
  print("Código executado com sucesso")
  


        