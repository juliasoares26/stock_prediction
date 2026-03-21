import sys
from pathlib import Path

MARGINALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MARGINALS_DIR))

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from arch import arch_model
import warnings
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew

from evt_gpd import GPD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiParametricGARCH_EVT:
  def __init__(self):
    self.garch_model = None
    self.garch_result = None
    self.residuals = None
    self.standardized_residuals = None

    self.gpd_left = None
    self.gpd_right = None

    self.threshold_left = None
    self.threshold_right = None

    self.diagnostics = {}
    self.tail_properties = {}

    self.fitted = False


  def _validate_gpd_params(self, xi, sigma, side='left'):
    """Valida parâmetros GPD para evitar valores extremos"""
    
    # ξ muito negativo causa upper endpoint muito próximo do threshold
    if xi < -0.9:
        logger.warning(f"ξ {side} muito negativo ({xi:.4f}), clipping para -0.9")
        xi = -0.9
    
    # ξ positivo muito alto causa caudas infinitas
    if xi > 0.5:
        logger.warning(f"ξ {side} muito positivo ({xi:.4f}), clipping para 0.5")
        xi = 0.5
    
    # σ deve ser positivo e razoável
    if sigma <= 0:
        raise ValueError(f"σ {side} deve ser positivo, obtido: {sigma}")
    
    # Para ξ < 0, verifica upper endpoint
    if xi < 0:
        upper_endpoint = -sigma / xi
        if upper_endpoint < 2.0:  # muito baixo
            logger.warning(f"Upper endpoint {side} muito baixo: {upper_endpoint:.4f}")
            # Reajusta ξ para endpoint = 3.0
            xi = -sigma / 3.0
    
    return xi, sigma

  def fit(
    self,
    returns: pd.Series,
    garch_spec: str = 'GARCH',
    garch_p: int = 1,
    garch_q: int = 1,
    dist: str = 'skewt',
    threshold_method: str = 'quantile',
    left_quantile: float = 0.05,
    right_quantile: float = 0.95,
    run_diagnostics: bool = True,
    plot_diagnostics: bool = False,
) -> Dict:

    logger.info("Ajustando modelo semi-paramétrico GARCH-EVT")
    
    # validação prévia dos dados
    if len(returns) < 500:
        raise ValueError(f"Amostra muito pequena: {len(returns)} obs (mínimo 500)")
    
    if returns.std() < 0.001:
        raise ValueError(f"Volatilidade quase nula: {returns.std():.6f}")
    
    if np.abs(returns.skew()) > 20:
        logger.warning(f"Skewness extrema detectada: {returns.skew():.2f}")
        logger.warning("Aplicando transformação robusta...")
        # Winsorizar retornos brutos antes do GARCH
        q01 = returns.quantile(0.001)
        q99 = returns.quantile(0.999)
        returns = returns.clip(q01, q99)
    
    # Armazena os retornos originais para uso posterior
    self._original_returns = returns.copy()
    self.returns = returns.copy()
    
    # Armazena os quantis para PIT 
    self.quantile_left = left_quantile
    self.quantile_right = right_quantile
    
    # Ajusta GARCH
    logger.info("Ajustando GARCH...")
    self._fit_garch_adaptive(returns, dist)
    
    # Extrai resíduos
    logger.info("Extraindo resíduos padronizados...")
    self._extract_residuals()
    
    # Diagnóstico GARCH
    if run_diagnostics:
        logger.info("Executando diagnósticos GARCH...")
        self.diagnose_garch()
    
    # Diagnóstico das caudas
    if run_diagnostics:
        logger.info("Analisando propriedades das caudas...")
        self.diagnose_tails(plot=plot_diagnostics)
    
    # Seleciona thresholds
    logger.info("Selecionando thresholds...")
    if threshold_method == 'hill':
        self._select_thresholds_hill()
    else:
        self._select_thresholds_auto(left_quantile, right_quantile)
    
    # Ajusta GPD
    logger.info("Ajustando GPD para ambas as caudas...")
    self._fit_tail_gpd()
    
    self._fitted = True
    self.fitted = True
    
    logger.info("Modelo ajustado com sucesso!")
    
    return self.get_summary()

  def cdf(self, x: float) -> float:
    """
    Calcula a Cumulative Distribution Function para um valor x usando o modelo semi-paramétrico GARCH-EVT 
    
    Parametros:
    x: float, valor do retorno para calcular a probabilidade acumulada
        
    Retornos:
    float - Probabilidade P(X <= x), entre 0 e 1
    """
    if not self._fitted:
        raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
    
    # Padroniza o retorno usando GARCH
    z = self._standardize_return(x)
    
    # Calcula CDF usando o modelo híbrido (GARCH core + GPD tails)
    return self._hybrid_cdf(z)

  def _standardize_return(self, x: float, t: int = None) -> float:
    """
    Converte retorno em resíduo padronizado usando volatilidade condicional
    """
    if t is None:
        # Fallback: usa volatilidade média
        mean = self.returns.mean()
        vol = np.sqrt(self.garch_result.conditional_volatility.mean())
    else:
        # Usa volatilidade condicional do tempo t
        mean = self.returns.mean()
        vol = self.garch_result.conditional_volatility.iloc[t]
    
    z = (x - mean) / vol
    return z

  def _hybrid_cdf(self, z: float) -> float:
    """
    CDF híbrida: usa distribuição GARCH no centro e GPD nas caudas
    
    Parametros
    z: float, Resíduo padronizado
        
    Returnos:
    float, Probabilidade acumulada
    """
    u_left  = self.threshold_left  if self.threshold_left  is not None else -np.inf
    u_right = self.threshold_right if self.threshold_right is not None else  np.inf
    zeta_l = self.quantile_left
    zeta_u = 1.0 - self.quantile_right

    if z <= u_left and self.gpd_left is not None:
        excess = u_left - z
        xi  = self.gpd_left.xi
        sig = self.gpd_left.sigma
        if xi < 0:
            max_excess = -sig / xi
            excess = min(excess, max_excess * 0.9999)
        if abs(xi) < 1e-8:
            gpd_sf = np.exp(-excess / sig)
        else:
            gpd_sf = max((1.0 + xi * excess / sig) ** (-1.0 / xi), 0.0)
        return float(np.clip(zeta_l * gpd_sf, 0.0, 1.0))

    elif z >= u_right and self.gpd_right is not None:
        excess = z - u_right
        xi  = self.gpd_right.xi
        sig = self.gpd_right.sigma
        if xi < 0:
            max_excess = -sig / xi
            excess = min(excess, max_excess * 0.9999)
        if abs(xi) < 1e-8:
            gpd_sf = np.exp(-excess / sig)
        else:
            gpd_sf = max((1.0 + xi * excess / sig) ** (-1.0 / xi), 0.0)
        return float(np.clip(1.0 - zeta_u * gpd_sf, 0.0, 1.0))

    else:
        resid = self.std_residuals
        center_mask = (resid >= u_left) & (resid <= u_right)
        center_vals  = resid[center_mask]
        if len(center_vals) == 0:
            return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))
        emp_cdf = np.mean(center_vals <= z)
        return float(np.clip(zeta_l + (1.0 - zeta_l - zeta_u) * emp_cdf, 0.0, 1.0))
    
  def probability_integral_transform(self, returns=None):
    if not hasattr(self, '_fitted') or not self._fitted:
        raise RuntimeError("Modelo precisa ser ajustado antes de usar PIT.")

    resid = self.std_residuals  # shape (n_train,)
    n = len(resid)

    # Ranks empíricos diretos: u_i = rank(z_i) / (n+1)
    # Isso produz range [1/(n+1), n/(n+1)] ≈ [0.0002, 0.9998]
    ranks = np.argsort(np.argsort(resid))  # rank 0-based
    u = (ranks + 1) / (n + 1)

    return np.clip(u, 1e-6, 1 - 1e-6)
  def _semi_parametric_cdf(self, z: float) -> float:
    """
    CDF semi-paramétrica: GARCH no centro + GPD nas caudas
    """
    # Cauda esquerda: z < threshold_left
    if z < self.threshold_left:
        # Proporção de dados na cauda esquerda
        p_left = self.left_quantile
        
        # Excedência sobre o threshold
        excess = self.threshold_left - z
        
        # CDF da GPD esquerda
        if self.gpd_left_xi != 0:
            gpd_cdf = 1 - (1 + self.gpd_left_xi * excess / self.gpd_left_sigma) ** (-1/self.gpd_left_xi)
        else:
            gpd_cdf = 1 - np.exp(-excess / self.gpd_left_sigma)
        
        # Escala para [0, p_left]
        return p_left * (1 - gpd_cdf)
    
    # Cauda direita: z > threshold_right
    elif z > self.threshold_right:
        # Proporção de dados na cauda direita
        p_right = 1 - self.right_quantile
        
        # Excedência sobre o threshold
        excess = z - self.threshold_right
        
        # CDF da GPD direita
        if self.gpd_right_xi != 0:
            gpd_cdf = 1 - (1 + self.gpd_right_xi * excess / self.gpd_right_sigma) ** (-1/self.gpd_right_xi)
        else:
            gpd_cdf = 1 - np.exp(-excess / self.gpd_right_sigma)
        
        # Escala para [p_right, 1]
        return self.right_quantile + p_right * gpd_cdf
    
    # Centro: usa CDF empírica dos resíduos padronizados
    else:
        # Conta quantos resíduos são <= z
        empirical_cdf = np.mean(self.std_residuals <= z)
        
        # Escala para [p_left, p_right]
        return (self.left_quantile + 
                (self.right_quantile - self.left_quantile) * empirical_cdf)
                
  def _fit_garch(self, returns: pd.Series, spec: str, p: int, q: int, dist: str):
    try:
      if spec.upper() == 'GARCH':
        vol = 'GARCH'
      elif spec.upper() == 'EGARCH':
        vol = 'EGARCH'
      elif spec.upper() in ['GJR', 'GJR-GARCH']:
        vol = 'GARCH'
        o = 1 
      else:
        raise ValueError(f"spec desconhecido: {spec}")
      
      if spec.upper() in ['GJR', 'GJR-GARCH']:
        self.garch_model = arch_model(
          returns,
          vol=vol,
          p=p,
          o=1,
          q=q,
          dist=dist,
          rescale=True
        )
      else:
        self.garch_model = arch_model(
          returns,
          vol=vol,
          p=p,
          q=q,
          dist=dist,
          rescale=True
      )

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.garch_result = self.garch_model.fit(disp='off',
                                                  show_warning=False,
                                                  options={'ftol': 1e-10})
  
      logger.info(f"Garch({p}, {q}) ajustado com dist={dist}")
      logger.info(f"AIC: {self.garch_result.aic:.2f}")
      logger.info(f"BIC: {self.garch_result.bic:.2f}")

    except Exception as e:
      logger.error(f"Erro ao ajustar GARCH: {str(e)}")
      raise
    
  def _fit_garch_adaptive(self, returns: pd.Series, dist: str = 'skewt'):
    
      specs_to_test = [
        ('GARCH', 1, 0, 1),   # GARCH(1,1) baseline
        ('GARCH', 2, 0, 1),   # GARCH(2,1) mais persistência
        ('GARCH', 1, 1, 1),   # GJR-GARCH(1,1) assimetria
        ('EGARCH', 1, 0, 1),  # EGARCH(1,1) leverage effect
    ]
    
      best_bic = np.inf
      best_result = None
      best_spec = None
    
      for vol_type, p, o, q in specs_to_test:
        try:
            if vol_type == 'GARCH' and o > 0:
                model = arch_model(returns, vol='GARCH', p=p, o=o, q=q, 
                                   dist=dist, rescale=True)
            else:
                model = arch_model(returns, vol=vol_type, p=p, q=q, 
                                   dist=dist, rescale=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp='off', show_warning=False, 
                                   options={'ftol': 1e-10})
            
            # Testa autocorrelação nos resíduos
            resid_std = result.resid / result.conditional_volatility
            resid_sq = resid_std ** 2
            lb_result = acorr_ljungbox(resid_sq, lags=10, return_df=True)
            n_sig = (lb_result['lb_pvalue'] < 0.05).sum()
            
            # Penaliza BIC se autocorrelação forte
            bic_adjusted = result.bic + (n_sig * 50)  # penalidade
            
            logger.info(f"  {vol_type}({p},{o},{q}): BIC={result.bic:.2f}, "
                       f"autocorr_lags={n_sig}/10")
            
            if bic_adjusted < best_bic:
                best_bic = bic_adjusted
                best_result = result
                best_spec = (vol_type, p, o, q)
                
        except Exception as e:
            logger.warning(f"  {vol_type}({p},{o},{q}) falhou: {e}")
    
      if best_result is None:
        raise ValueError("Todas especificações GARCH falharam")
    
      self.garch_result = best_result
      logger.info(f"✓ Melhor modelo: {best_spec[0]}({best_spec[1]},{best_spec[2]},{best_spec[3]})")
      logger.info(f"  BIC={best_result.bic:.2f}, AIC={best_result.aic:.2f}")
    
      return best_result
  
  def _extract_residuals(self) -> None:
    self._extract_standardized_residuals()
  
  def _extract_standardized_residuals(self) -> None:
    resid = self.garch_result.resid / self.garch_result.conditional_volatility
    
    # Repadronizar 
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)
    resid = (resid - resid_mean) / resid_std
    
    # Detectar outliers
    p1, p99 = np.percentile(resid, [1, 99])
    iqr_robust = p99 - p1
    lower_bound = p1 - 3 * iqr_robust
    upper_bound = p99 + 3 * iqr_robust
    
    n_outliers = np.sum((resid < lower_bound) | (resid > upper_bound))
    pct_outliers = 100 * n_outliers / len(resid)
    
    # Clipping apenas se >5% de outliers
    if pct_outliers > 5:
        logger.warning(f"Clipping {n_outliers} outliers extremos ({pct_outliers:.1f}%)")
        lower_bound = np.percentile(resid, 0.1)
        upper_bound = np.percentile(resid, 99.9)
        resid = np.clip(resid, lower_bound, upper_bound)
        
        # Re-padronizar após clipping
        resid = (resid - np.mean(resid)) / np.std(resid)
    elif n_outliers > 0:
        logger.info(f"Clipping {n_outliers} outliers extremos")
        resid = np.clip(resid, lower_bound, upper_bound)
        resid = (resid - np.mean(resid)) / np.std(resid)
    
    # Validação final 
    final_std = np.std(resid)
    final_skew = abs(skew(resid))
    
    if final_std < 0.9 or final_std > 1.1:
        logger.warning(f"Std final fora do ideal: {final_std:.4f} (esperado ~1.0)")
    
    if final_skew > 3:
        logger.warning(f"Skewness alta após padronização: {final_skew:.4f}")
    
    self.standardized_residuals = pd.Series(resid, index=self.garch_result.resid.index)
    logger.info(f"Resíduos extraídos: média={np.mean(resid):.4f}, std={np.std(resid):.4f}")

  def _select_thresholds_auto(self, left_quantile: float = 0.05, right_quantile: float = 0.95):
    resid = self.standardized_residuals.values
    
    self.threshold_left = np.quantile(resid, left_quantile)
    self.threshold_right = np.quantile(resid, right_quantile)
    
    n_left = np.sum(resid < self.threshold_left)
    n_right = np.sum(resid > self.threshold_right)
    
    logger.info(f"threshold left (q={left_quantile}): {self.threshold_left:.4f} ({n_left} obs)")
    logger.info(f"threshold right (q={right_quantile}): {self.threshold_right:.4f} ({n_right} obs)")

  def _select_thresholds_hill(self):
    resid = self.standardized_residuals.values

    left_data = -resid[resid < 0]
    left_sorted = np.sort(left_data)[::-1]
    n_left = len(left_sorted)

    k_min, k_max = max(10, int(0.05 * n_left)), int(0.20 * n_left)
    k_values = np.arange(k_min, k_max)
    hill_estimates = []

    for k in k_values:
      log_ratios = np.log(left_sorted[:k]) - np.log(left_sorted[k])
      hill_estimates.append(np.mean(log_ratios))

    best_k_idx = np.argmin(np.abs(np.array(hill_estimates) - 0.2))
    optimal_k_left = k_values[best_k_idx]
    self.threshold_left = left_sorted[optimal_k_left]

    right_data = resid[resid > 0]
    right_sorted = np.sort(right_data)[::-1]
    n_right = len(right_sorted)

    k_min, k_max = max(10, int(0.05 * n_right)), int(0.20 * n_right)
    k_values = np.arange(k_min, k_max)
    hill_estimates = []

    for k in k_values:
      log_ratios = np.log(right_sorted[:k]) - np.log(right_sorted[k])
      hill_estimates.append(np.mean(log_ratios))

    best_k_idx = np.argmin(np.abs(np.array(hill_estimates) - 0.2))
    optimal_k_right = k_values[best_k_idx]
    self.threshold_right = right_sorted[optimal_k_right]

    logger.info(f"Hill threshold left: {self.threshold_left:.4f} (k={optimal_k_left})")
    logger.info(f"Hill threshold right: {self.threshold_right:.4f} (k={optimal_k_right})")

  def _fit_tail_gpd(self):
    resid = self.standardized_residuals.values

    # cauda esquerda
    left_mask = resid < self.threshold_left
    n_left = np.sum(left_mask)

    if n_left < 30:
        logger.warning(f"Poucas excedências na cauda esquerda ({n_left}), usando empírico")
        self.gpd_left = None
    else:
        left_exceedances = np.abs(resid[left_mask] - self.threshold_left)
        
        # Verificar qualidade das excedências
        if len(np.unique(left_exceedances)) < 10:
            logger.warning("Excedências esquerdas com pouca variabilidade, usando empírico")
            self.gpd_left = None
        else:
            self.gpd_left = GPD()
            try:
                result_left = self.gpd_left.fit(
                    left_exceedances,
                    threshold=0,
                    method='mle',
                    return_std_errors=True
                )

                # Validar resultado
                xi_left = result_left['xi']
                sigma_left = result_left['sigma']
                
                # Rejeitar se convergiu para bound artificial ou parâmetros inválidos
                if abs(xi_left - (-0.5)) < 0.001:  # Detecta convergência para bound
                    logger.warning(f"ξ esquerda convergiu para bound ({xi_left:.4f}), refitando com PWM")
                    # Tentar método alternativo (Probability Weighted Moments)
                    try:
                        result_left = self.gpd_left.fit(
                            left_exceedances,
                            threshold=0,
                            method='pwm',
                            return_std_errors=False
                        )
                        xi_left = result_left['xi']
                        sigma_left = result_left['sigma']
                        logger.info(f"PWM esquerda bem-sucedido: ξ={xi_left:.4f}")
                    except Exception as e_pwm:
                        logger.warning(f"PWM esquerda falhou ({str(e_pwm)}), usando empírico")
                        self.gpd_left = None
                        xi_left = None
                
                # Logging final corrigido 
                if xi_left is not None:
                    if abs(xi_left) > 0.8:
                        logger.warning(f"ξ esquerda extremo ({xi_left:.4f}), usando empírico")
                        self.gpd_left = None
                    elif sigma_left <= 0 or sigma_left > 100:
                        logger.warning(f"σ esquerda inválido ({sigma_left:.4f}), usando empírico")
                        self.gpd_left = None
                    else:
                        # Sucesso = logar resultados
                        logger.info("GPD cauda esquerda:")
                        xi_se = result_left.get('xi_se')
                        sigma_se = result_left.get('sigma_se')
                        
                        if xi_se is not None:
                            logger.info(f"ξ = {xi_left:.4f} ± {xi_se:.4f}")
                        else:
                            logger.info(f"ξ = {xi_left:.4f}")
                        
                        if sigma_se is not None:
                            logger.info(f"σ = {sigma_left:.4f} ± {sigma_se:.4f}")
                        else:
                            logger.info(f"σ = {sigma_left:.4f}")
                        
                        logger.info(f"  Exceedances: {n_left}")
                        
            except Exception as e:
                logger.error(f"Erro GPD esquerda: {e}, usando empírico")
                self.gpd_left = None

    # Cauda Direita
    right_mask = resid > self.threshold_right
    n_right = np.sum(right_mask)

    if n_right < 30:
        logger.warning(f"Poucas excedências na cauda direita ({n_right}), usando empírico")
        self.gpd_right = None
    else:
        right_exceedances = resid[right_mask] - self.threshold_right
        
        # Verificar qualidade das excedências
        if len(np.unique(right_exceedances)) < 10:
            logger.warning("Excedências direitas com pouca variabilidade, usando empírico")
            self.gpd_right = None
        else:
            self.gpd_right = GPD()
            try:
                result_right = self.gpd_right.fit(
                    right_exceedances,
                    threshold=0,
                    method='mle',
                    return_std_errors=True
                )

                # Validar resultado
                xi_right = result_right['xi']
                sigma_right = result_right['sigma']
                
                # Rejeitar se convergiu para bound artificial
                if abs(xi_right - (-0.5)) < 0.001:
                    logger.warning(f"ξ direita convergiu para bound ({xi_right:.4f}), refitando com PWM")
                    try:
                        result_right = self.gpd_right.fit(
                            right_exceedances,
                            threshold=0,
                            method='pwm',
                            return_std_errors=False
                        )
                        xi_right = result_right['xi']
                        sigma_right = result_right['sigma']
                        logger.info(f"PWM direita bem-sucedido: ξ={xi_right:.4f}")
                    except Exception as e_pwm:
                        logger.warning(f"PWM direita falhou ({str(e_pwm)}), usando empírico")
                        self.gpd_right = None
                        xi_right = None
                
                # Logging final 
                if xi_right is not None:
                    if abs(xi_right) > 0.8:
                        logger.warning(f"ξ direita extremo ({xi_right:.4f}), usando empírico")
                        self.gpd_right = None
                    elif sigma_right <= 0 or sigma_right > 100:
                        logger.warning(f"σ direita inválido ({sigma_right:.4f}), usando empírico")
                        self.gpd_right = None
                    else:
                        # Sucesso - logar resultados
                        logger.info("GPD cauda direita:")
                        xi_se = result_right.get('xi_se')
                        sigma_se = result_right.get('sigma_se')
                        
                        if xi_se is not None:
                            logger.info(f"  ξ = {xi_right:.4f} ± {xi_se:.4f}")
                        else:
                            logger.info(f"  ξ = {xi_right:.4f}")
                        
                        if sigma_se is not None:
                            logger.info(f"  σ = {sigma_right:.4f} ± {sigma_se:.4f}")
                        else:
                            logger.info(f"  σ = {sigma_right:.4f}")
                        
                        logger.info(f"  Exceedances: {n_right}")
                        
            except Exception as e:
                logger.error(f"Erro GPD direita: {e}, usando empírico")
                self.gpd_right = None

    MIN_SIGMA = 0.3

# Valida GPD esquerda
    if self.gpd_left is not None:
    # Rejeita σ muito pequeno OU ξ positivo com σ < 0.5
      if (self.gpd_left.sigma < MIN_SIGMA or 
        (self.gpd_left.xi > 0 and self.gpd_left.sigma < 0.5)):
        logger.warning(
            f"GPD esquerda problemática (ξ={self.gpd_left.xi:.4f}, "
            f"σ={self.gpd_left.sigma:.4f}), usando empírico"
        )
        self.gpd_left = None

# Valida GPD direita
    if self.gpd_right is not None:
      if (self.gpd_right.sigma < MIN_SIGMA or 
        (self.gpd_right.xi > 0 and self.gpd_right.sigma < 0.5)):
        logger.warning(
            f"GPD direita problemática (ξ={self.gpd_right.xi:.4f}, "
            f"σ={self.gpd_right.sigma:.4f}), usando empírico"
        )
        self.gpd_right = None
    if self.gpd_left is not None:
        self.gpd_left_xi = self.gpd_left.xi
        self.gpd_left_sigma = self.gpd_left.sigma
    else:
        self.gpd_left_xi = None
        self.gpd_left_sigma = None
    
    # Extrair e armazenar parâmetros da GPD direita
    if self.gpd_right is not None:
        self.gpd_right_xi = self.gpd_right.xi
        self.gpd_right_sigma = self.gpd_right.sigma
    else:
        self.gpd_right_xi = None
        self.gpd_right_sigma = None
    
    # Armazenar resíduos padronizados 
    self.std_residuals = resid


  def _validate_and_fix_gpd(self):
    """
    Valida e corrige parâmetros GPD extremos
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Validação GPD Esquerda
    if self.gpd_left is not None:
        xi_left = self.gpd_left.xi
        sigma_left = self.gpd_left.sigma
        
        # σ muito pequeno (< 0.1)
        if sigma_left < 0.1:
            logger.warning(f"GPD Left: σ muito pequeno ({sigma_left:.4f}), ajustando para 0.5")
            sigma_left = 0.5
        
        # ξ muito negativo (< -0.9)
        if xi_left < -0.9:
            logger.warning(f"GPD Left: ξ muito negativo ({xi_left:.4f}), clipping para -0.8")
            xi_left = -0.8
        
        # ξ muito positivo (> 0.5)
        if xi_left > 0.5:
            logger.warning(f"GPD Left: ξ muito positivo ({xi_left:.4f}), clipping para 0.5")
            xi_left = 0.5
        
        # Para ξ < 0, verifica upper endpoint razoável
        if xi_left < 0:
            endpoint = -sigma_left / xi_left
            if endpoint < 1.5:
                logger.warning(f"GPD Left: endpoint muito baixo ({endpoint:.2f}), reajustando")
                # Força endpoint = 3.0
                xi_left = -sigma_left / 3.0
        
        # Atualiza GPD se houve mudanças
        if abs(xi_left - self.gpd_left.xi) > 1e-6 or abs(sigma_left - self.gpd_left.sigma) > 1e-6:
            logger.info(f"Recriando GPD esquerda: ξ={xi_left:.4f}, σ={sigma_left:.4f}")
            from evt_gpd import GPD
            self.gpd_left = GPD()
            self.gpd_left.xi = xi_left
            self.gpd_left.sigma = sigma_left
            self.gpd_left.threshold = self.threshold_left
            self.gpd_left.n_exceedances = len(self.returns[self.returns < self.threshold_left])
    
    # Validação GPD Direita
    if self.gpd_right is not None:
        xi_right = self.gpd_right.xi
        sigma_right = self.gpd_right.sigma
        
        # σ muito pequeno
        if sigma_right < 0.1:
            logger.warning(f"GPD Right: σ muito pequeno ({sigma_right:.4f}), ajustando para 0.5")
            sigma_right = 0.5
        
        # ξ muito negativo
        if xi_right < -0.9:
            logger.warning(f"GPD Right: ξ muito negativo ({xi_right:.4f}), clipping para -0.8")
            xi_right = -0.8
        
        # ξ muito positivo
        if xi_right > 0.5:
            logger.warning(f"GPD Right: ξ muito positivo ({xi_right:.4f}), clipping para 0.5")
            xi_right = 0.5
        
        # Endpoint 
        if xi_right < 0:
            endpoint = -sigma_right / xi_right
            if endpoint < 1.5:
                logger.warning(f"GPD Right: endpoint muito baixo ({endpoint:.2f}), reajustando")
                xi_right = -sigma_right / 3.0
        
        # Atualiza GPD
        if abs(xi_right - self.gpd_right.xi) > 1e-6 or abs(sigma_right - self.gpd_right.sigma) > 1e-6:
            logger.info(f"Recriando GPD direita: ξ={xi_right:.4f}, σ={sigma_right:.4f}")
            from evt_gpd import GPD
            self.gpd_right = GPD()
            self.gpd_right.xi = xi_right
            self.gpd_right.sigma = sigma_right
            self.gpd_right.threshold = self.threshold_right
            self.gpd_right.n_exceedances = len(self.returns[self.returns > self.threshold_right])
            
  def diagnose_tails(self, plot: bool = True) -> Dict:
    if self.standardized_residuals is None:
      raise ValueError("Execute _extract_residuals primeiro")
    
    resid = self.standardized_residuals.values

    diagnostics = {}

    _, p_shapiro = stats.shapiro(resid[:5000] if len(resid) > 5000 else resid)
    diagnostics['shapiro_pvalue'] = p_shapiro

    diagnostics['kurtosis_excess'] = stats.kurtosis(resid)
    diagnostics['skewness'] = stats.skew(resid)

    left_data = -resid[resid < 0]
    right_data = resid[resid > 0]

    if len(left_data) > 50:
      left_sorted = np.sort(left_data)[::-1]
      k = min(100, len(left_sorted) // 4)
      hill_left = np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
      diagnostics['hill_xi_left'] = hill_left

    if len(right_data) > 50:
      right_sorted = np.sort(right_data)[::-1]
      k = min(100, len(right_sorted) // 4)
      hill_right = np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
      diagnostics['hill_xi_right'] = hill_right

    if 'hill_xi_left' in diagnostics:
      if diagnostics['hill_xi_left'] > 0.5:
        tail_type_left = 'muito pesada - fréchet'
      elif diagnostics['hill_xi_left'] > 0.1:
        tail_type_left = "pesada - pareto"
      elif diagnostics['hill_xi_left'] > -0.1:
          tail_type_left = "exponencial - Gumbel"
      else:
          tail_type_left = "limitada - Weibull"
      diagnostics['tail_type_left'] = tail_type_left

    if 'hill_xi_right' in diagnostics:
      if diagnostics['hill_xi_right'] > 0.5:
        tail_type_right = 'muito pesada - fréchet'
      elif diagnostics['hill_xi_right'] > 0.1:
        tail_type_right = "pesada - pareto"
      elif diagnostics['hill_xi_right'] > -0.1:
          tail_type_right = "exponencial - Gumbel"
      else:
          tail_type_right = "limitada - Weibull"
      diagnostics['tail_type_right'] = tail_type_right

    xi_left = diagnostics.get('hill_xi_left', 0)
    xi_right = diagnostics.get('hill_xi_right', 0)

    if xi_left > 0.2 and xi_right > 0.2:
      copula_suggestion = 't-student - tail-dependence simetrica'
    elif xi_left > 0.2 and xi_right < 0.1:
      copula_suggestion = "clayton - lower tail dependence"
    elif xi_left < 0.1 and xi_right > 0.2:
      copula_suggestion = "gumbel ou joe - upper tail dependence"
    else:
      copula_suggestion = "gaussian - sem tail dependence"
    
    diagnostics['copula_suggestion'] = copula_suggestion

    self.tail_properties = diagnostics

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        stats.probplot(resid, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title('QQ-Plot (Normal)')
        
        axes[0, 1].hist(resid, bins=50, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(resid.min(), resid.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), 'r-', lw=2)
        axes[0, 1].set_title('Distribuição dos Resíduos')
        axes[0, 1].set_xlabel('Resíduos Padronizados')
        
        if len(left_data) > 50:
            left_sorted = np.sort(left_data)[::-1]
            k_range = range(10, min(200, len(left_sorted) // 2))
            hill_estimates = [np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
                            for k in k_range]
            axes[0, 2].plot(k_range, hill_estimates)
            axes[0, 2].axhline(y=diagnostics['hill_xi_left'], color='r', linestyle='--')
            axes[0, 2].set_title('Hill Plot - Cauda Esquerda')
            axes[0, 2].set_xlabel('k')
            axes[0, 2].set_ylabel('ξ estimado')
        
        if len(right_data) > 50:
            right_sorted = np.sort(right_data)[::-1]
            k_range = range(10, min(200, len(right_sorted) // 2))
            hill_estimates = [np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
                            for k in k_range]
            axes[1, 0].plot(k_range, hill_estimates)
            axes[1, 0].axhline(y=diagnostics['hill_xi_right'], color='r', linestyle='--')
            axes[1, 0].set_title('Hill Plot - Cauda Direita')
            axes[1, 0].set_xlabel('k')
            axes[1, 0].set_ylabel('ξ estimado')
        
        if len(left_data) > 50:
            thresholds = np.percentile(left_data, np.linspace(80, 98, 20))
            mean_excess = [np.mean(left_data[left_data > u] - u) for u in thresholds]
            axes[1, 1].plot(thresholds, mean_excess, 'o-')
            axes[1, 1].set_title('Mean Excess Plot - Esquerda')
            axes[1, 1].set_xlabel('Threshold u')
            axes[1, 1].set_ylabel('Mean Excess')
        
        if len(right_data) > 50:
            thresholds = np.percentile(right_data, np.linspace(80, 98, 20))
            mean_excess = [np.mean(right_data[right_data > u] - u) for u in thresholds]
            axes[1, 2].plot(thresholds, mean_excess, 'o-')
            axes[1, 2].set_title('Mean Excess Plot - Direita')
            axes[1, 2].set_xlabel('Threshold u')
            axes[1, 2].set_ylabel('Mean Excess')
        
        plt.tight_layout()
        plt.savefig('tail_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info("Diagnóstico de Caudas")
    logger.info(f"Kurtosis excess: {diagnostics['kurtosis_excess']:.4f}")
    logger.info(f"Skewness: {diagnostics['skewness']:.4f}")
    
    if 'hill_xi_left' in diagnostics:
        logger.info(f"Hill ξ esquerda: {diagnostics['hill_xi_left']:.4f}")
    else:
        logger.info("Hill ξ esquerda: N/A")
    
    if 'hill_xi_right' in diagnostics:
        logger.info(f"Hill ξ direita: {diagnostics['hill_xi_right']:.4f}")
    else:
        logger.info("Hill ξ direita: N/A")
    
    logger.info(f"Tipo cauda esquerda: {diagnostics.get('tail_type_left', 'N/A')}")
    logger.info(f"Tipo cauda direita: {diagnostics.get('tail_type_right', 'N/A')}")
    logger.info(f"Sugestão de copula: {diagnostics['copula_suggestion']}")
    
    if 'hill_xi_left' in diagnostics and 'hill_xi_right' in diagnostics:
        left_sorted = np.sort(-resid[resid < 0])[::-1]
        right_sorted = np.sort(resid[resid > 0])[::-1]
        
        k_range = range(30, min(150, len(left_sorted)//3))
        hill_left_estimates = [np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
                               for k in k_range]
        hill_right_estimates = [np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
                                for k in k_range]
        
        stability_left = np.std(hill_left_estimates[-50:])
        stability_right = np.std(hill_right_estimates[-50:])
        
        diagnostics['hill_stability_left'] = stability_left
        diagnostics['hill_stability_right'] = stability_right
        
        if stability_left > 0.1 or stability_right > 0.1:
            logger.warning("Hill plots NÃO estáveis (σ > 0.1)")
        else:
            logger.info("Hill plots estáveis")
    
    return diagnostics

  def diagnose_garch(self) -> Dict:
    if self.standardized_residuals is None:
      raise ValueError("ajuste o modelo primeiro")
    
    resid_sq = self.standardized_residuals ** 2

    lb_result = acorr_ljungbox(resid_sq, lags=20, return_df=True)

    n_significant = (lb_result['lb_pvalue'] < 0.05).sum()
    autocorr_present = n_significant > 10

    diagnostics = {
      'ljungbox_results': lb_result,
      'autocorr_detected': autocorr_present,
      'n_significant_lags': n_significant,
      'mean_pvalue': lb_result['lb_pvalue'].mean()
    }

    self.diagnostics['garch'] = diagnostics

    if autocorr_present:
      logger.warning("Autocorrelação detectada nos resíduos ao quadrado!")
    else:
        logger.info("sem autocorrelação significativa nos resíduos²")
    
    return diagnostics
  
  def predict_volatility(self, horizon: int = 1) -> np.ndarray:
    if not self.fitted:
      raise ValueError("ajuste o modelo primeiro")
    
    forecast = self.garch_result.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.values[-1, :]) 
  
  def var(self, p: float, horizon: int = 1) -> float:
    if not self.fitted:
        raise ValueError("Ajuste o modelo primeiro")

    sigma_forecast = self.predict_volatility(horizon)[0]
    mu_forecast = self.garch_result.params.get('mu', 0) if 'mu' in self.garch_result.params else 0

    resid = self.standardized_residuals.values
    n = len(resid)
    
    n_left = np.sum(resid < self.threshold_left)
    prob_left_tail = n_left / n

    if (1 - p) <= prob_left_tail and self.gpd_left is not None:
        conditional_prob = (1 - p) / prob_left_tail
        
        xi = self.gpd_left.xi
        sigma_gpd = self.gpd_left.sigma
        
        if abs(xi) < 1e-6:
            gpd_exceedance = sigma_gpd * (-np.log(1 - conditional_prob))
        else:
            gpd_exceedance = (sigma_gpd / xi) * ((1 - conditional_prob) ** (-xi) - 1)
        
        var_residual = self.threshold_left - gpd_exceedance
    else:
        var_residual = np.quantile(resid, 1 - p)
    
    var_return = mu_forecast + sigma_forecast * var_residual
    
    return var_return
  
  def expected_shortfall(self, p: float, horizon: int = 1) -> float:
    if not self.fitted:
        raise ValueError("Ajuste o modelo primeiro")
    
    sigma_forecast = self.predict_volatility(horizon)[0]
    mu_forecast = self.garch_result.params.get('mu', 0) if 'mu' in self.garch_result.params else 0

    resid = self.standardized_residuals.values
    n = len(resid)
    
    n_left = np.sum(resid < self.threshold_left)
    prob_left_tail = n_left / n

    if (1 - p) <= prob_left_tail and self.gpd_left is not None:
        conditional_prob = (1 - p) / prob_left_tail
        
        xi = self.gpd_left.xi
        sigma_gpd = self.gpd_left.sigma
        
        if abs(xi) < 1e-6:
            gpd_quantile = sigma_gpd * (-np.log(1 - conditional_prob))
        else:
            gpd_quantile = (sigma_gpd / xi) * ((1 - conditional_prob) ** (-xi) - 1)
        
        if abs(xi) < 1e-6:
            es_exceedance = gpd_quantile + sigma_gpd
        elif xi < 1:
            es_exceedance = (gpd_quantile + sigma_gpd) / (1 - xi)
        else:
            logger.warning(f"ES não existe para ξ={xi:.4f} >= 1, usando VaR+σ")
            es_exceedance = gpd_quantile + sigma_gpd
        
        es_residual = self.threshold_left - es_exceedance
    else:
        var_threshold = np.quantile(resid, 1 - p)
        es_residual = resid[resid < var_threshold].mean()
    
    es_return = mu_forecast + sigma_forecast * es_residual
    
    return es_return
  
  def get_summary(self) -> Dict:
    summary = {
      'garch': {
        'aic': self.garch_result.aic,
        'bic': self.garch_result.bic,
        'params': self.garch_result.params.to_dict()
      }
    }

    if self.gpd_left is not None:
      summary['gpd_left'] = {
        'xi': self.gpd_left.xi,
        'sigma': self.gpd_left.sigma,
        'threshold': self.threshold_left,
        'n_exceedances': self.gpd_left.n_exceedances
      }
    
    if self.gpd_right is not None:
      summary['gpd_right'] = {
        'xi': self.gpd_right.xi,
        'sigma': self.gpd_right.sigma,
        'threshold': self.threshold_right,
        'n_exceedances': self.gpd_right.n_exceedances
      }

    return summary