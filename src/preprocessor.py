import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Tuple, List, Optional, Dict
import logging
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor:

  def __init__(self,
               lookback_window: int = 60,
               prediction_horizon: int = 1,
               target_type: str = 'returns',
               normalization: str = 'rolling',
               scaler_type: str = 'robust'
  ):
    
    self.lookback_window = lookback_window
    self.prediction_horizon = prediction_horizon
    self.target_type = target_type
    self.normalization = normalization
    self.scaler_type = scaler_type

    if scaler_type == 'minmax':
      self.scaler_class = MinMaxScaler
    elif scaler_type == 'standard':
      self.scaler_class = StandardScaler
    elif scaler_type == 'robust':
      self.scaler_class = RobustScaler
    else:
      raise ValueError(f"Scaler '{scaler_type}' não reconhecido")

    self.global_scaler = None
    self.feature_columns = None

  def create_returns_target(self, prices: np.ndarray, horizon: int = 1) -> np.ndarray:

    returns = np.log(prices[horizon:] / prices[:-horizon])
    returns = np.concatenate([np.full(horizon, np.nan), returns])
    return returns
  
  def create_direction_target(self, prices: np.ndarray, horizon: int = 1) -> np.ndarray:
    returns = self.create_returns_target(prices, horizon)
    direction = (returns > 0).astype(int)
    return direction

  def create_classification_target(
      self,
      prices: np.darray,
      horizon: int = 1,
      thresholds: Tuple[float, float] = (-0.01, 0.01)
  ) -> np.ndarray:
    
    returns = self.create_returns_target(prices, horizon)

    classes = np.ones_like(returns)
    classes[returns < thresholds[0]] = 0
    classes[returns > thresholds[1]] = 2

    return classes.astype(int)
  
  def normalize_rolling_window(
      self,
      data: pd.DataFrame,
      window: int = 252,
  ) -> pd.DataFrame:
    
    normalized = pd.DataFrame(index=data.index,columns=data.columns)

    for col in data.columns:
      rolling_mean = data[col].rolling(window=window, min_periods=1).mean()
      rolling_std = data[col].rolling(window=window, min_periods=1).std()

      normalized[col] = (data[col] - rolling_mean) / (rolling_std + 1e-8)

    return normalized

  def normalize_expanding_window(self, data: pd.DataFrame) -> pd.DataFrame:

    normalized = pd.DataFrame(index=data.index, columns=data.columns)

    for col in data.columns:
      expanding_mean = data[col].expanding(min_periods=1).mean()
      expanding_std = data[col].expanding(min_periods=1).std()

      normalized[col] = (data[col] - expanding_mean) / (expanding_std + 1e-8)

    return normalized
  