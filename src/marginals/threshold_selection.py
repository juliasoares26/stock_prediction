import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdSelector:
  def __init__(self, data: np.ndarray):
    self.data = np.sort(data)[::-1]
    self.n = len(data)

  def mean_excess_function(
      self,
      thresholds: Optional[np.ndarray] = None,
      min_exceedances: int = 50
  ) -> pd.DataFrame:
    
    if thresholds is None:
      quantiles = np.linspace(0.50, 0.99, 50)
      thresholds = np.quantile(self.data, quantiles)

    results = []

    for u in thresholds:
      exceedances = self.data[self.data > u] - u
      n_exceed = len(exceedances)

      if n_exceed >= min_exceedances:
        mean_excess = np.mean(exceedances)
        std_excess = np.std(exceedances) / np.sqrt(n_exceed)

        results.append({
          'threshold': u,
          'mean_excess': mean_excess,
          'se': std_excess,
          'ci_lower': mean_excess - 1.96 * std_excess,
          'ci_upper': mean_excess + 1.96 * std_excess,
          'n_exceedances': n_exceed
        })
    
    df = pd.DataFrame(results)
    logger.info(f"MEF calculado para {len(df)} thresholds")

    return df
  
  def parameter_stability_plot(
      self,
      thresholds: Optional[np.ndarray] = None,
      min_exceedances: int = 50
  ) -> pd.DataFrame:
    import sys
    sys.path.append('..')

    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evt_gpd import GPD

    if thresholds is None:
      quantiles = np.linspace(0.70, 0.98, 30)
      thresholds = np.quantile(self.data, quantiles)

    results = []

    for u in thresholds:
      exceedances = self.data[self.data > u]
      n_exceed = len(exceedances)

      if n_exceed < min_exceedances:
        continue

      try:
        gpd = GPD()
        fit_results = gpd.fit(self.data, u, method='mle', return_std_errors=True)

        results.append({
          'threshold': u,
          'xi': fit_results['xi'],
          'sigma': fit_results['sigma'],
          'xi_se':fit_results.get('xi_se', np.nan),
          'sigma_se': fit_results.get('sigma_se', np.nan),
          'n_exceedances': n_exceed,
          'aic': fit_results['aic']
        }) 
      except:
        continue

    df = pd.DataFrame(results)
    logger.info(f"Parameter stability calculado para {len(df)} thresholds")

    return df
  
  def hill_estimator(
      self,
      k_range: Optional[Tuple[int, int]] = None
  ) -> pd.DataFrame:
    
    if k_range is None:
      k_min = max(10, int(0.01 * self.n))
      k_max = int(0.3 * self.n)
    else:
      k_min, k_max = k_range

    k_values = range(k_min, min(k_max, self.n-1))
    results = []

    sorted_data = np.sort(self.data)[::-1]

    for k in k_values:
      if k >= len(sorted_data):
        break

      log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
      xi_hill = np.mean(log_ratios)

      xi_se = xi_hill / np.sqrt(k)

      threshold = sorted_data[k]

      results.append({
        'k': k,
        'threshold': threshold,
        'xi_hill': xi_hill,
        'xi_se':xi_se,
        'ci_lower': xi_hill - 1.96 * xi_se,
        'ci_upper': xi_hill + 1.96 * xi_se
      })

    df = pd.DataFrame(results)
    logger.info(f"hill estimator calculado para k ∈ [{k_min}, {k_max}]")

    return df
  
  def automated_selection(
      self,
      method: str = 'stability',
      min_exceedances: int = 50
  ) -> Dict:
    logger.info(f" seleção automática via método: {method}")

    if method == 'stability':
      return self._select_by_stability(min_exceedances)
    elif method == 'mef':
      return self._select_by_mef(min_exceedances)
    elif method == 'aic':
      return self._select_by_aic(min_exceedances)
    elif method == 'combined':
      return self._select_combined(min_exceedances)
    else:
      raise ValueError(f"método desconhecido: {method}")
    
  def _select_by_stability(self, min_exceedances:int) -> Dict:
    df = self.parameter_stability_plot(min_exceedances=min_exceedances)

    if len(df) < 5:
      logger.warning("Dados insuficientes para seleção automaica")
      return {'threshold': np.quantile(self.data, 0.90), 'method': 'default'}

    # variancia rolling dos parametros
    window = min(5, len(df) // 3)
    df['xi_var'] = df['xi'].rolling(window, center=True).std()

    # threshold onde a var é minima - estabilidade
    idx_stable = df['xi_var'].idxmin()
    selected = df.loc[idx_stable]

    logger.info(f"threshold selecionado: {selected['threshold']:.4f} ({selected['n_exceedances']} exceedances)")

    return {
      'threshold': selected['threshold'],
      'xi': selected['xi'],
      'sigma': selected['sigma'],
      'n_exceedances': selected['n_exceedances'],
      'method': 'stability'
    }
  
  # seleciona o threshold onde MEF começa a ser linear
  def _select_by_mef(self, min_exceedances: int) -> Dict:
  
    df = self.mean_excess_function(min_exceedances=min_exceedances)

    if len(df) < 10:
      return {'threshold': np.quantile(self.data, 0.90), 'method': 'default'}
    
    # seleciona threshold onde MEF começa a ser linear
    window = min(10, len(df) // 2)
    candidates = []

    for i in range(len(df) - window):
      segment = df.iloc[i:i+window]
      x = segment['threshold'].values
      y = segment['mean_excess'].values

      # regressão linear
      slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)
     
      residuals = y - (slope * x + intercept)
      _, normality_p = stats.shapiro(residuals) if len(residuals) >= 3 else (0,1)
      # threshold onde R² é máximo 

    if i > 0:
      prev_slope = candidates[-1]['slope'] if candidates else 0
      slope_change = abs(slope - prev_slope)
    else:
      slope_change = 0

    candidates.append({
      'index': i,
      'r_squared': r_value ** 2,
      'p_value': p_value,
      'slope': slope,
      'slope_positive': slope > 0,
      'normality_p': normality_p,
      'slope_change': slope_change
    })

    valid = [c for c in candidates if c['slope_positive'] and
             c['p_value'] < 0.05 and
             c['normality_p'] > 0.05 and
             c['r_squared'] > 0.8]
    
    if valid:
      selected_idx = min(valid, key=lambda c: c['slope_change'])['index']

    else:
      selected_idx = max(candidates, key=lambda c: c['r_squared'])['index']

    selected = df.iloc[selected_idx]
    
    return {
      'threshold': selected['threshold'],
      'n_exceedances': selected['n_exceedances'],
      'method': 'mef'
    }
  
  def _select_by_aic(self, min_exceedances: int) -> Dict:
    df = self.parameter_stability_plot(min_exceedances=min_exceedances)

    if len(df) == 0:
      return {'threshold': np.quantile(self.data, 0.90), 'method': 'default'}
    
    idx_best = df['aic'].idxmin()
    selected = df.loc[idx_best]

    logger.info(f"threshold AIC: {selected['threshold']:.4f}")

    return {
      'threshold': selected['threshold'],
      'xi': selected['xi'],
      'sigma': selected['sigma'],
      'n_exceedances': selected['n_exceedances'],
      'aic': selected['aic'],
      'method': 'aic'
    }

  def _select_combined(self, min_exceedances: int) -> Dict:
    stability = self._select_by_stability(min_exceedances)
    aic = self._select_by_aic(min_exceedances)
    mef = self._select_by_mef(min_exceedances)

    thresholds = [
        stability.get('threshold', np.nan),
        aic.get('threshold', np.nan),
        mef.get('threshold', np.nan)
    ]

    valid_thresholds = [t for t in thresholds if not np.isnan(t)]
    
    if len(valid_thresholds) < 2:
        logger.warning("Apenas 1 threshold válido encontrado")
        return {
            'threshold': valid_thresholds[0] if valid_thresholds else np.quantile(self.data, 0.90),
            'method': 'single',
            'stability': stability,
            'aic': aic,
            'mef': mef
        }

    # filtragem de outilers para poucos dados
    
    # Usando Median Absolute Deviation que é mais robusto
    median = np.median(valid_thresholds)
    mad = np.median([abs(t - median) for t in valid_thresholds])
    
    # Critério adaptativo baseado no número de métodos
    if len(valid_thresholds) == 2:
        # Com 2 métodos, só remover se extremamente discrepante
        # 5 * MAD 
        k = 5.0
    elif len(valid_thresholds) == 3:
        # Com 3 métodos, moderadamente conservador
        # 3 * MAD
        k = 3.0
    else:
        # Com mais métodos, ser mais rigoroso
        k = 2.5
    
    # MAD = 0 todos os valores são iguais
    if mad == 0:
        filtered = valid_thresholds
        logger.info("Todos os thresholds são idênticos")
    else:
        # Remover apenas outliers extremos
        lower_bound = median - k * mad
        upper_bound = median + k * mad
        
        filtered = [t for t in valid_thresholds 
                   if lower_bound <= t <= upper_bound]
        
        # Se filtrar demais, relaxar
        if len(filtered) < 2:
            logger.warning(f"Filtro muito restritivo (k={k}), relaxando para k=5")
            lower_bound = median - 5.0 * mad
            upper_bound = median + 5.0 * mad
            filtered = [t for t in valid_thresholds 
                       if lower_bound <= t <= upper_bound]
        
        # Última tentativa, aceitar pelo menos o mais próximo da mediana
        if len(filtered) == 0:
            filtered = [min(valid_thresholds, key=lambda t: abs(t - median))]
            logger.warning("Nenhum threshold passou no filtro, usando o mais próximo da mediana")
    
    if len(filtered) < len(valid_thresholds):
        removed = [t for t in valid_thresholds if t not in filtered]
        logger.warning(f"Removidos outliers: {[f'{r:.2f}' for r in removed]}")

    
    # Preparar métodos e seus metadados
    method_data = []
    
    # estabilidade
    if stability['threshold'] in filtered:
        xi_se = stability.get('xi_se', np.nan)
        if not np.isnan(xi_se) and xi_se > 0:
            weight = 1.0 / xi_se  # Menor erro padrão = maior peso
        else:
            weight = 1.0  # Peso padrão se não houver erro padrão
        
        method_data.append({
            'threshold': stability['threshold'],
            'weight': weight,
            'method': 'stability'
        })
    
    # AIC
    if aic['threshold'] in filtered:
        aic_value = aic.get('aic', np.inf)
        xi_se = aic.get('xi_se', np.nan)
        
        # Peso baseado em AIC e erro padrão
        if aic_value != np.inf and not np.isnan(xi_se) and xi_se > 0:
            # Menor AIC e menor erro padrão = maior peso
            weight = np.exp(-aic_value / 1000) / xi_se  # Normalizar AIC
        else:
            weight = 0.5  # Peso reduzido se não houver metadados
        
        method_data.append({
            'threshold': aic['threshold'],
            'weight': weight,
            'method': 'aic'
        })
    
    # MEF
    if mef['threshold'] in filtered:
        n_exceed = mef.get('n_exceedances', 0)
        
        # Peso baseado no número de exceedances
        if n_exceed > 0:
            weight = np.sqrt(n_exceed) / 10  # Normalizar
        else:
            weight = 0.5
        
        method_data.append({
            'threshold': mef['threshold'],
            'weight': weight,
            'method': 'mef'
        })
    
    
    if len(method_data) >= 2:
        # Normalizar pesos
        total_weight = sum(m['weight'] for m in method_data)
        for m in method_data:
            m['weight'] /= total_weight
        
        # Média ponderada
        combined_threshold = sum(m['threshold'] * m['weight'] for m in method_data)
        
        logger.info(f"Usando média ponderada com {len(method_data)} métodos")
        for m in method_data:
            logger.info(f"  {m['method']}: {m['threshold']:.4f} (peso={m['weight']:.3f})")
    
    elif len(method_data) == 1:
        combined_threshold = method_data[0]['threshold']
        logger.info(f"Apenas 1 método após filtragem: {method_data[0]['method']}")
    
    else:
        # Fallback: usar mediana de todos os válidos
        combined_threshold = np.median(valid_thresholds)
        logger.warning("Nenhum método sobreviveu à filtragem, usando mediana de todos")
    

#avaliando consistencia
    
    consistency = self._assess_threshold_consistency(valid_thresholds)
    
    if not consistency['consistent']:
        logger.warning(
            f"Métodos apresentam baixa consistência "
            f"(CV={consistency['cv']:.2f}, range={consistency['range']:.2f})"
        )
        
        # Se consistência muito baixa, priorizar o método mais conservador
        if consistency['cv'] > 0.5:
            logger.warning("CV > 0.5: priorizando threshold mais conservador (menor)")
            combined_threshold = min(filtered)
    
    logger.info(f"Thresholds válidos: {[f'{t:.4f}' for t in valid_thresholds]}")
    logger.info(f"Thresholds após filtro: {[f'{t:.4f}' for t in filtered]}")
    logger.info(f"Threshold combinado final: {combined_threshold:.4f}")
    logger.info(f"Consistência: CV={consistency['cv']:.2f}")

    return {
        'threshold': combined_threshold,
        'method': 'combined',
        'consistency': consistency,
        'stability': stability,
        'aic': aic,
        'mef': mef,
        'filtered_thresholds': filtered
    }
  
  def _assess_threshold_consistency(self, thresholds: List[float]) -> Dict:
    valid = [t for t in thresholds if t > 0]

    if len(valid) < 2:
      return {'consistent': False, 'cv': np.inf}
    
    mean_thresh = np.mean(valid)
    std_thresh = np.std(valid)
    cv = std_thresh / mean_thresh

    return {
      'consistent': cv < 0.3,
      'cv': cv,
      'range': max(valid) - min(valid),
      'mean': mean_thresh
    }
  
  def plot_diagnostics(self, save_path: Optional[str] = None):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))


    # mean excess function
    mef_df = self.mean_excess_function()
    ax = axes[0, 0]
    ax.plot(mef_df['threshold'], mef_df['mean_excess'], 'b-', label='MEF')
    ax.fill_between(
      mef_df['threshold'],
      mef_df['ci_lower'],
      mef_df['ci_upper'],
      alpha=0.3,
      label='95% CI'
        )
    ax.set_xlabel('Threshold u')
    ax.set_ylabel('Mean Excess e(u)')
    ax.set_title('Mean Excess Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
        
    # Estabilidade - Xi
    stab_df = self.parameter_stability_plot()
    ax = axes[0, 1]
    ax.errorbar(
    stab_df['threshold'],
      stab_df['xi'],
      yerr=1.96 * stab_df['xi_se'],
      fmt='o-',
      label='ξ estimate'
        )
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold u')
    ax.set_ylabel('Shape ξ')
    ax.set_title('Parameter Stability - Shape')
    ax.legend()
    ax.grid(True, alpha=0.3)
        
    # Estabilidade - Sigma
    ax = axes[1, 0]
    ax.errorbar(
      stab_df['threshold'],
      stab_df['sigma'],
      yerr=1.96 * stab_df['sigma_se'],
      fmt='o-',
      color='green',
      label='σ estimate'
      )
    ax.set_xlabel('Threshold u')
    ax.set_ylabel('Scale σ')
    ax.set_title('Parameter Stability - Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
        
    # Hill Plot
    hill_df = self.hill_estimator()
    ax = axes[1, 1]
    ax.plot(hill_df['k'], hill_df['xi_hill'], 'b-', label='Hill estimate')
    ax.fill_between(
      hill_df['k'],
      hill_df['ci_lower'],
      hill_df['ci_upper'],
      alpha=0.3,
      label='95% CI'
        )
    ax.set_xlabel('k (order statistics)')
    ax.set_ylabel('ξ Hill')
    ax.set_title('Hill Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
        
    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')
      logger.info(f"Diagnósticos salvos em {save_path}")
        
    return fig
  
if __name__ == "__main__":

  # teste
  np.random.seed(42)

  # dados com cauda pesada - pareto
  alpha = 2.5
  data = (np.random.pareto(alpha, 5000) + 1) * 10

  selector = ThresholdSelector(data)

  print("threshold selectot")

  result = selector.automated_selection(method='combined', min_exceedances=50)

  print(f"threshold selecionado: {result['threshold']:.4f}")
  print(f"método: {result['method']}")

  selector.plot_diagnostics('threshold_diagnostics.png')