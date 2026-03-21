import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy import stats, special
from scipy.linalg import sqrtm
import warnings

def standardize(data: np.array, method: str = 'zscore') -> np.ndarray:

  if method == 'zscore':
    return (data - np.mean(data)) / np.std(data)
  
  elif method == 'minmax':
    return (data - np.min(data)) / (np.max(data) - np.min(data))
  
  elif method == 'robust':
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return (data - median) / (1.4826 * mad)
  
  else:
    raise ValueError(f"Método desconhecido: {method}")
  
def rank_transform(data: np.ndarray) -> np.ndarray:
  n = len(data)
  ranks = stats.rankdata(data)
  return ranks / (n+1)

def inverse_rank_transform(ranks: np.ndarray, original_data: np.ndarray) -> np.ndarray:
  sorted_data = np.sort(original_data)
  n = len(sorted_data)
  indices = np.clip(ranks * n, 0, n - 1).astype(int)
  return sorted_data[indices]

# matrizes e algelin

def ensure_positive_definite(
    matrix: np.ndarray,
    method: str = 'eigenvalue',
    epsilon: float = 1e-8
) -> np.ndarray:
  
  if method == 'eigenvalue':
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    eigenvalues = np.maximum(eigenvalues, epsilon)

    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
  
  elif method == 'cholesky':
    try:
      np.linalg.cholesky(matrix)
      return matrix
    except np.linalg.LinAlgError:
      return matrix + np.eye(matrix.shape[0]) * epsilon
    
  elif method == 'nearPD':
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)

    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
      return A3
    
    spacing = np.spacing(np.linalg.norm(A3))
    I = np.eye(A3.shape[0])
    k = 1
    while not is_positive_definite(A3):
      mineig = np.min(np.real(np.linalg.eigvals(A3)))
      A3 += I * (-mineig * k**2 + spacing)
      k += 1

    return A3
  else:
    raise ValueError(f"Método desconhecido: {method}")
  
def is_positive_definite(matrix: np.ndarray) -> bool:
  try:
    np.linalg.cholesky(matrix)
    return True
  except np.linalg.LinAlgError:
    return False
  
def correlation_to_covariance(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
  D = np.diag(std)
  return D @ corr @ D

def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
  std = np.sqrt(np.diag(cov))
  D_inv = np.diag(1 / std)
  return D_inv @ cov @ D_inv

def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
  return sqrtm(matrix).real
    
# estatisticas descritivas

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
  if abs(denominator) < 1e-10:
    return default
  return numerator / denominator

def skewness(data: np.ndarray, bias: bool = True) -> float:
  return stats.skew(data, bias=bias)

def kurtosis(data: np.ndarray, excess: bool=True) -> float:
  return stats.kurtosis(data, fisher=excess)

def jarque_bera_test(data: np.ndarray) -> Tuple[float, float]:
  jb_stat, p_value = stats.jarque_bera(data)
  return jb_stat, p_value

# quantis e percentis

def quantile_weighted(data: np.ndarray, weights: np.ndarray, q: float) -> float:
  sorted_idx = np.argsort(data)
  sorted_data = data[sorted_idx]
  sorted_weights = weights[sorted_idx]

  cumsum = np.cumsum(sorted_weights)
  cumsum = cumsum / cumsum[-1]

  idx = np.searchsorted(cumsum, q)
  if idx >= len(sorted_data):
    return sorted_data[-1]
  return sorted_data[idx]

def conditional_quantile(
    y: np.ndarray,
    x: np.ndarray,
    x_value: float,
    q: float,
    bandwidth: Optional[float] = None
) -> float:
  
  if bandwidth is None:
    #regra de silverman
    bandwidth = 1.06 * np.std(x) * len(x) ** (-1/5)

  weights = np.exp(-0.5 * ((x - x_value) / bandwidth) ** 2)
  weights = weights / weights.sum()

  return quantile_weighted(y, weights, q)

# distribuições

def studentt_pdf(x: np.ndarray, df:float, loc: float = 0, scale: float = 1) -> np.ndarray:
  return stats.t.pdf(x, df, loc=loc, scale=scale)

def studentt_cdf(x: np.ndarray, df: float, loc: float = 0, scale: float = 1) -> np.ndarray:
  return stats.t.cdf(x, df, loc=loc, scale=scale)

def multivariate_studentt_pdf(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    df: float
) -> float:
  d = len(mean)
  x_centered = x - mean

  numerator = special.gamma((df + d) / 2)
  denominator = (
    special.gamma(df / 2)*
    (df * np.pi) ** (d/2) *
    np.sqrt(np.linalg.det(cov))
  )

  cov_inv = np.linalg.inv(cov)
  mahalanobis = x_centered @ cov_inv @ x_centered

  kernel = (1 + mahalanobis / df) ** (-(df + d) / 2)

  return (numerator / denominator) * kernel

# otimização numérica

def numerical_gradient(
    func: callable,
    x: np.ndarray,
    epsilon: float = 1e-18
) -> np.ndarray:
  
  grad = np.zeros_like(x)

  for i in range(len(x)):
    x_plus = x.copy()
    x_minus = x.copy()

    x_plus[i] += epsilon
    x_minus[i] -= epsilon

    grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)

  return grad

def numerical_hessian(
    func: callable,
    x: np.ndarray,
    epsilon: float = 1e-5
) -> np.ndarray:
  
  n = len(x)
  hessian = np.zeros((n, n))

  for i in range(n):
    for j in range(n):
      x_pp = x.copy()
      x_pm = x.copy()
      x_mp = x.copy()
      x_mm = x.copy()

      x_pp[i] += epsilon
      x_pp[j] += epsilon

      x_pm[i] += epsilon
      x_pm[j] -= epsilon

      x_mp[i] -= epsilon
      x_mp[j] += epsilon

      x_mm[i] -= epsilon
      x_mm[j] -= epsilon

      hessian[i, j] = (
        func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)
      ) / (4 * epsilon ** 2)

  return hessian

# séries temporais

def autocorrelation(data: np.array, lag: int = 1) -> float:
  if lag >= len(data) or lag <1:
    raise ValueError(f"lag deve ser entre 1 e {len(data)-1}")
  return np.corrcoef(data[:-lag], data[lag:])[0,1]

def ljung_box_test(data: np.ndarray, lags: int = 10) -> Tuple[float, float]:
  from statsmodels.stats.diagnostic import acorr_ljungbox
  result = acorr_ljungbox(data, lags=lags, return_df=False)
  return result[0][-1], result[1][-1]

def detrend(data: np.ndarray, method: str = 'linear') -> np.ndarray:
  from scipy.signal import detrend as scipy_detrend
  if method == 'mean':
    return data - np.mean(data)
  else:
    return scipy_detrend(data, type=method)
  
# utilidades

def winsorize(data: np.ndarray, limits: Tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
  lower = np.quantile(data, limits[0])
  upper = np.quantile(data, limits[1])
  return np.clip(data, lower, upper)

def rolling_apply(
    data: np.ndarray,
    window: int,
    func: callable,
    min_periods: Optional[int] = None
) -> np.ndarray:
  
  if min_periods is None:
    min_periods = window

  result = np.full(len(data), np.nan)

  for i in range(window - 1, len(data)):
    window_data = data[i - window + 1:i + 1]

    if len(window_data[~np.isnan(window_data)]) >= min_periods:
      result[i] = func(window_data[~np.isnan(window_data)])

  return result

def expanding_apply(data: np.ndarray, func: callable, min_periods: int = 1) -> np.ndarray:
  result = np.full(len(data), np.nan)

  for i in range(min_periods - 1, len(data)):
    window_data = data[:i + 1]

    if len(window_data[~np.isnan(window_data)]) >= min_periods:
      result[i] = func(window_data[~np.isnan(window_data)])
  
  return result

if __name__ == "__main__":

  # testes

  # teste de matriz positive definite

  corr = np.array([
    [1.0, 0.8, 0.5],
    [0.8, 1.0, 0.6],
    [0.5, 0.6, 1.0]
  ])

  print("matriz de correlação")
  print(corr)
  print(f"é positive definite? {is_positive_definite(corr)}")

  # rank transform
  data = np.random.rand(1000)
  uniform = rank_transform(data)

  print(f"rank transform:")
  print(f"Original: min={data.min():.2f}, max={data.max():.2f}")
  print(f"Uniforme: min={uniform.min():.4f}, max={uniform.max():.4f}")

  # teste de estatisticas
  print(f"estatisticas descritivas:")
  print(f"skewness: {skewness(data):.4f}")
  print(f"curtose: {kurtosis(data):.4f}")

  jb_stat, jb_pval = jarque_bera_test(data)
  print(f"jarque-bera: stat={jb_stat:.2f}, p-value={jb_pval:.4f}")
