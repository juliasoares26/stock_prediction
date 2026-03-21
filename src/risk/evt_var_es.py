import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import logging
from scipy import stats
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger =  logging.getLogger(__name__)

class EVTRiskMetrics:

  def __init__(self):
    self.marginal_models = {} # modelos evt por ativo
    self.copula_model = None
    self.portfolio_weights = None
    self.asset_names = None
    self.marginal_cdfs = {} # cdfs semi-parametricas
    self.marginal_inv_cdfs = {} 


  # define modelos marginais evt para cada ativo
  def set_marginal_models(
      self,
      marginal_models: Dict,
      asset_names: Optional[List[str]] = None
  ):
    
    self.marginal_models = marginal_models

    if asset_names is None:
      self.asset_names = list(marginal_models.keys())
    else:
      self.asset_names = asset_names

    self._precompute_marginal_transforms()

    logger.info(f"modelos marginais definidos: {len(self.marginal_models)} ativos")
  
  # define modelo de cópula para dependencia multivariada
  # copula model: instancia da copula fitted - GaussianCopula, StudentCopula, VineCopula
  # copula type: gaussian, t-student, clayton, gumbel, vine

  def set_copula_model(
      self,
      copula_model,
      copula_type: str = 'gaussian'
  ):
    
    self.copula_model = copula_model
    self.copula_type = copula_type

    logger.info(f"copula definida: {copula_type}")

  # pré computa cdfs semi-parametricas e suas inversas
  def _precompute_marginal_transforms(self):

    logger.info("pré-computando transformações marginais")

    for asset_name, model in self.marginal_models.items():
      # retornos para interpolação
      returns_grid = np.linspace(
        model.returns.min() - 3*model.returns.std(),
        model.returns.max() + 3*model.returns.std(),
        1000
      )

      # cdf semi parametrica - empirica no centro e gpd nas caudas
      cdf_values = np.array([
        self._semi_parametric_cdf(r, model)
        for r in returns_grid
      ])

      # interpoladores
      self.marginal_cdfs[asset_name] = interp1d(
        returns_grid, cdf_values,
        kind='linear',
        bounds_error=False,
        fill_value=(0.0, 1.0)
      )

      self.marginal_inv_cdfs[asset_name] = interp1d(
        cdf_values, returns_grid,
        kind='linear',
        bounds_error=False,
        fill_value=(returns_grid[0], returns_grid[-1])
      )

      logger.info("transformações pré computadas")

  def _semi_parametric_cdf(
      self,
      x: float,
      model   
  ) -> float:
    
    # thresholds 
    threshold_lower = np.quantile(model.returns, 0.05)
    threshold_upper = np.quantile(model.returns, 0.95)

      # cauda esquerda - usa gpd fitted na cauda inferior
    if x < threshold_lower:
      if hasattr(model, 'gpd_left') and model.gpd_left is not None:
        excess = threshold_lower - x  # Cauda esquerda
        cdf_excess = model.gpd_left.cdf(excess)
        return 0.05 * (1 - cdf_excess)
      else:
        return np.mean(model.returns <= x)
      
    elif x > threshold_upper:
      if hasattr(model, 'gpd_right') and model.gpd_right is not None:
        excess = x - threshold_upper
        cdf_excess = model.gpd_right.cdf(excess)
        return 0.95 + 0.05 * cdf_excess
      else:
        return np.mean(model.returns <= x)
    
    # Centro: CDF empírica
    else:
      return np.mean(model.returns <= x)
      
  # transforma uniforme [0,1] em retornos via inversa da cdf marginal
  def _transform_uniform_to_returns(
      self, 
      uniform_samples: np.ndarray,
      asset_name: str
  ) -> np.ndarray:
    
    inv_cdf = self.marginal_inv_cdfs[asset_name]
    return inv_cdf(uniform_samples)
  
  def portfolio_var(
      self,
      weights: np.ndarray, # peso do portfolio
      confidence_level: float = 0.95,
      n_simulations: int = 10000, # num de simulações monte carlo
      horizon: int = 1, # horizonte em dias
      method: str = 'copula',
      copula_type: Optional[str] = None,
      seed: Optional[int] = None
  ) -> float:
    
    if len(weights) != len(self.asset_names):
      raise ValueError(f"numero de pesos {len(weights)} != assets {len(self.asset_names)}")

    if not np.isclose(weights.sum(), 1.0):
      logger.warning(f"soma dos pesos {weights.sum():.4f}, normalizando")

    self.portfolio_weights = weights

    logger.info(f"calculando portfolio VaR {confidence_level:.0%} (h={horizon})")

    if method == 'copula':
      return self._var_copula_simulation(
        weights, confidence_level, n_simulations, horizon, copula_type, seed
      )
    
    elif method == 'independence':
      return self._var_independence_simulation(
        weights, confidence_level, n_simulations, horizon, seed
      )
    
    elif method == 'analytical':
      return self._var_analytical(weights, confidence_level, horizon)
    else:
      raise ValueError(f"metodo desconhecio: {method}")
    

  # VaR via simulação monte carlo com copulas - capturar dependencia multivariada
  def _var_copula_simulation(
      self, 
      weights: np.ndarray,
      confidence_level: float, 
      n_simulations: int,
      horizon: int,
      copula_type: Optional[str],
      seed: Optional[int]
  ) -> float:
    
      if self.copula_model is None:
        raise ValueError("Copula não definida, use set_copula_model() primeiro")
      if seed is not None:
        np.random.seed(seed)

      # simula uniformes correlacionadas via cópula
      if hasattr(self.copula_model, 'simulate'):
        uniform_samples = self.copula_model.simulate(
          n_samples = n_simulations,
          seed=seed
        )
      else:
        if copula_type == 'gaussian' or copula_type is None:
          uniform_samples = self._simulate_gaussian_copula(n_simulations, seed)
        elif copula_type == 't-student':
          uniform_samples = self._simulate_t_copula(n_simulations, seed)
        else:
          raise ValueError(f"Copula type {copula_type} não implementada")
        
      # transforma uniformes em retornos via cdf marginal inversa sem simular independencia
      simulated_returns = np.zeros((n_simulations, len(self.asset_names)))

      for i, asset_name in enumerate(self.asset_names):
        u_samples = uniform_samples[:, i]

        simulated_returns[:, i] = self._transform_uniform_to_returns(
          u_samples, asset_name
        )

      # ajusta para horizonte multi-periodo
      if horizon > 1:
        simulated_returns *= np.sqrt(horizon)

      portfolio_returns = simulated_returns @ weights

      var = np.quantile(portfolio_returns, 1 - confidence_level)

      logger.info(f" VaR {confidence_level:.0%} (copula): {var:.4f}")

      return var
  
  def _simulate_gaussian_copula(
      self,
      n_simulations: int,
      seed: Optional[int]
  ) -> np.ndarray:
    
    if seed is not None:
      np.random.seed(seed)

    # estima matriz de correlação dos retornos
    returns_matrix = pd.DataFrame({
      asset: model.returns
      for asset, model in self.marginal_models.items()
    })    

    corr_matrix = returns_matrix.corr().values

    # simula normais multivariadas
    mean = np.zeros(len(self.asset_names))
    z_samples = np.random.multivariate_normal(
      mean, corr_matrix, size=n_simulations
    )  


    uniform_samples = stats.norm.cdf(z_samples)

    return uniform_samples
  
  # simula uniformes de copula t-student
  def _simulate_t_copula(
      self,
      n_simulations: int,
      seed: Optional[int],
      df: float = 5.0
  ) -> np.ndarray:
    
    if seed is not None:
      np.random.seed(seed)

    returns_matrix = pd.DataFrame({
      asset: model.returns
      for asset, model in self.marginal_models.items()
    })
    
    corr_matrix = returns_matrix.corr().values

    # simulando uma t-student multivariada
    mean = np.zeros(len(self.asset_names))

    #Z ~ N(0, Σ), S ~ χ²(df)
    #T = Z/sqrt(S/df) ~ t_df

    z_samples = np.random.multivariate_normal(
      mean, corr_matrix, size = n_simulations
    )

    chi2_samples = np.random.chisquare(df, size=n_simulations)
    t_samples = z_samples / np.sqrt(chi2_samples[:, None] / df)

    uniform_samples = stats.t.cdf(t_samples, df=df)

    return uniform_samples
  
  # VaR assumindo independencia
  def _var_independence_simulation(
      self,
      weights: np.ndarray,
      confidence_level: float,
      n_simulations: int,
      horizon: int,
      seed: Optional[int]
  ) -> float:
    
    if seed is not None:
      np.random.seed(seed)

    logger.warning("usando independencia")

    uniform_samples = np.random.uniform(0, 1, (n_simulations, len(self.asset_names)))

    simulated_returns = np.zeros((n_simulations, len(self.asset_names)))

    for i, asset_name in enumerate(self.asset_names):
      simulated_returns[:, i] = self._transform_uniform_to_returns(
        uniform_samples[:, i], asset_name
      )

    if horizon > 1:
      simulated_returns *= np.sqrt(horizon)

    portfolio_returns = simulated_returns @ weights
    var = np.quantile(portfolio_returns, 1 - confidence_level)

    logger.info(f"VaR {confidence_level:.0%} (independência): {var:.4f}")

    return var
  

  def _var_analytical(
      self,
      weights: np.ndarray,
      confidence_level: float,
      horizon: int
  ) -> float:
    
    marginal_vars = np.zeros(len(self.asset_names))

    for i, asset_name in enumerate(self.asset_names):
      model = self.marginal_models[asset_name]

      q = 1 - confidence_level
      marginal_vars[i] = np.quantile(model.returns, q)

      if horizon > 1:
        marginal_vars[i] *= np.sqrt(horizon)

    var_portfolio = weights @ marginal_vars

    logger.info(f"VaR {confidence_level:.0%} (analítico): {var_portfolio}")

    return var_portfolio
  
  def portfolio_es(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.95,
      n_simulations: int = 10000,
      horizon: int = 1,
      method: str = 'copula',
      seed: Optional[int] = None
  ) -> float:
    
    # expected shortfall do portfolio - es/cVaR
    # es = E[perda | perda > var]

    if seed is not None:
      np.random.seed(seed)

    logger.info(f"calculando portfolio es {confidence_level:.0%} (h={horizon})")

    # simulando retornos
    if method == 'copula':
      if self.copula_model is None:
        raise ValueError("cópula não definida")
      
      uniform_samples = self.copula_model.simulate(n_samples=n_simulations, seed=seed) \
        if hasattr(self.copula_model, 'simulate') \
        else self._simulate_gaussian_copula(n_simulations, seed)
      
    else:
      uniform_samples = np.random.uniform(0, 1, (n_simulations, len(self.asset_names)))

    simulated_returns = np.zeros((n_simulations, len(self.asset_names)))
    for i, asset_name in enumerate(self.asset_names):
      simulated_returns[:, i] = self._transform_uniform_to_returns(
        uniform_samples[:, i], asset_name
      )

    if horizon > 1:
      simulated_returns *= np.sqrt(horizon)

    portfolio_returns = simulated_returns @ weights

    # ES = média condicional além do VaR
    var = np.quantile(portfolio_returns, 1 - confidence_level)
    losses_beyond_var = portfolio_returns[portfolio_returns <= var]
    es = losses_beyond_var.mean()

    logger.info(f"ES {confidence_level:.0%}: {es:.4f}")
    logger.info(f"es/var ratio: {es/var:.3f}")

    return es
  
  # compara var/es vom diferentes estruturas de dependencia
  def compare_dependence_impact(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.99,
      n_simulations: int = 10000,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    
    logger.info(f"comparando impacto da estrutura de dependencia")

    results = []

    # independencia
    var_indep = self.portfolio_var(
      weights, confidence_level, n_simulations, 1, 'independence', seed=seed
    )
    es_indep = self.portfolio_es(
      weights, confidence_level, n_simulations, 1, 'independence', seed=seed
    )

    results.append({
      'Method': 'independence',
      'VaR': var_indep,
      'ES': es_indep,
      'ES/VAR': es_indep / var_indep
    })

    # copula gaussiana
    var_gauss = self.portfolio_var(
      weights, confidence_level, n_simulations, 1, 'copula', 'gaussian', seed
    )
    es_gauss = self.portfolio_es(
      weights, confidence_level, n_simulations, 1, 'copula', seed
    )

    results.append({
      'Method': 'gaussian copula',
      'VaR': var_gauss,
      'ES': es_gauss,
      'ES/VaR': es_gauss / var_gauss
    })

    # copula t-student
    var_t = self.portfolio_var(
      weights, confidence_level, n_simulations, 1, 'copula', 't-student', seed
    )
    es_t = self.portfolio_es(
      weights, confidence_level, n_simulations, 1, 'copula', seed
    )

    results.append({
        'Method': 't-Student Copula',
        'VaR': var_t,
        'ES': es_t,
        'ES/VaR': es_t / var_t
    })
    
    df = pd.DataFrame(results)

    # adiciona diferenças relativas
    df['VaR_vs_Indep_%'] = 100 * (df['VaR'] - var_indep) / abs(var_indep)
    df['ES_vs_Indep_%'] = 100 * (df['ES'] - es_indep) / abs(es_indep)

    logger.info(f"comparação concluida")
    logger.info(f"{df.to_string(index=False)}")

    return df
  
if __name__ == "__main__":
  print("modulo carregado com sucesso")

          

  





