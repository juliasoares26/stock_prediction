import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:

  def __init__(self, prices: pd.DataFrame):
    self.prices = prices.copy()
    self.returns = None
    self.log_returns = None
    self.clean_prices = None


  def handle_missing_data(
      self,
      method: str = "forward_fill",
      max_consecutive_missing: int = 5
  ) -> pd.DataFrame:
    logger.info(f"resolvendo a falta de dados usando: {method}")

    for col in self.prices.columns:
      consecutive_na = self.prices[col].isnull().astype(int).groupby(
        self.prices[col].notnull().astype(int).cumsum()
      ).sum()

      max_consecutive = consecutive_na.max() if len(consecutive_na) > 0 else 0

      if max_consecutive > max_consecutive_missing:
        logger.warning(
          f"Ticker {col} tem {max_consecutive} dias sem dados"
        )

      if method == "forward_fill":
        self.clean_prices = self.prices.ffill().bfill()

      elif method == "interpolate":
        self.clean_prices = self.prices.fillna(method='linear', limit=max_consecutive_missing)
        self.clean_prices = self.clean_prices.fillna(method='ffill').fillna(method='bfill')

      elif method == "drop":
        self.clean_prices = self.prices.dropna()
        logger.info(f"removemos {len(self.prices) - len(self.clean_prices)} colunas com valores faltantes")

      else:
        raise ValueError(f"metodo desconhecido: {method}")
      
      if self.clean_prices.isnull().sum().sum() > 0:
        logger.warning("ainda temos valores faltantes após a limpeza, vamos removê-los")
        self.clean_prices = self.clean_prices.dropna()

      logger.info(f"limpando: {self.clean_prices.shape}")
      return self.clean_prices
    
  def detect_outliers(
      self,
      method: str = "iqr",
      threshold: float = 3.0
  ) -> pd.DataFrame:

    if self.returns is None:
      self.compute_returns()

    outliers = pd.DataFrame(False, index=self.returns.index, columns=self.returns.columns)

    for col in self.returns.columns:
      series = self.returns[col].dropna()

      if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        IQR = q3 - q1
        lower = q1 - threshold * IQR
        upper = q3 + threshold * IQR
        outliers[col] = (series < lower) | (series > upper)

      elif method == "zscore":
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outliers[col] = z_scores > threshold

      elif method == "mad":
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / mad
        outliers[col] = np.abs(modified_z) > threshold

      else:
        raise ValueError(f"metodo desconhecido: {method}")
      
      n_outliers = outliers.sum().sum()
      logger.info(f"foram encontrados {n_outliers} outliers usando o método {method}")

      return outliers
    

  def winsorize_outliers(
      self,
      lower_percentile: float = 0.01,
      upper_percentile: float = 0.99,
  ) -> pd.DataFrame:
    
    if self.returns is None:
      self.compute_returns()

    winsorized = self.returns.copy()
    total_modified = 0

    for col in winsorized.columns:
      series = winsorized[col]
      lower_bound = series.quantile(lower_percentile)
      upper_bound = series.quantile(upper_percentile)

      winsorized[col] = series.clip(lower = lower_bound, upper= upper_bound)

      n_modified = ((series < lower_bound) | (series > upper_bound)).sum()
      total_modified += n_modified
      if n_modified > 0:
        logger.debug(f"{col}: {n_modified} valores winzorizados")

    logger.info(f"total de {total_modified} valores winzorizados em {len(winsorized.columns)}")
    return winsorized
  
  def compute_returns(
      self,
      method: str = "log",
      periods: int = 1
  ) -> pd.DataFrame:
    
    if self.clean_prices is None:
      self.handle_missing_data()

    if method == "simple":
      returns = self.clean_prices.pct_change(periods=periods)

    elif method == "log":
      returns = np.log(self.clean_prices / self.clean_prices.shift(periods))

    else:
      raise ValueError(f"metodo desconhecido: {method}")
    
    returns = returns.iloc[periods:]

    if method == "log":
      self.log_returns = returns
    else:
      self.returns = returns

    self.returns = returns

    logger.info(f"método {method}. shape: {returns.shape}")
    return returns

  def check_stationary(
      self,
      alpha: float = 0.05
  ) -> pd.DataFrame:
    
    from statsmodels.tsa.stattools import adfuller

    if self.returns is None:
      self.compute_returns()

    results = {
      'ticker': [],
      'adf_statistic': [],
      'p_value': [],
      'is_stationary': []
    }

    for col in self.returns.columns:
      series = self.returns[col].dropna()

      try:
        adf_result = adfuller(series, autolag='AIC')
        results['ticker'].append(col)
        results['adf_statistic'].append(adf_result[0])
        results['p_value'].append(adf_result[1])
        results['is_stationary'].append(adf_result[1] < alpha)
      except Exception as e:
        logger.warning(f"teste adf falhou para {col}: {str(e)}")
        results['ticker'].append(col)
        results['adf_statistic'].append(np.nan)
        results['p_value'].append(np.nan)
        results['is_stationary'].append(False)

    results_df = pd.DataFrame(results)
    n_stationary = results_df['is_stationary'].sum()
    logger.info(f"as séries {n_stationary}/len(results_df) são estacionárias no nível {alpha}")
    return results_df
  
  def get_descriptive_stats(self) -> pd.DataFrame:
    if self.returns is None:
      self.compute_returns()

    stats_dict = {
      'mean':self.returns.mean(),
      'std': self.returns.std(),
      'skewness': self.returns.skew(),
      'kurtosis': self.returns.kurtosis(),
      'min': self.returns.min(),
      'max': self.returns.max(),
      'q25': self.returns.quantile(0.25),
      'median': self.returns.median(),
      'q75': self.returns.quantile(0.75),
    }

    stats_dict['annual_return'] = self.returns.mean() * 252
    stats_dict['annual_volatility'] = self.returns.std() * np.sqrt(252)
    stats_dict['sharpe_ratio'] = stats_dict['annual_return'] / stats_dict['annual_volatility']
    
    jb_stats = []
    jb_pvals = []
    for col in self.returns.columns:
      series = self.returns[col].dropna()
      try: 
        jb_stat, jb_pval = stats.jarque_bera(series)
        jb_stats.append(jb_stat)
        jb_pvals.append(jb_pval)
      except:
        jb_stats.append(np.nan)
        jb_pvals.append(np.nan)

    stats_dict['jb_statistic'] = jb_stats
    stats_dict['jb_pvalue'] = jb_pvals
    stats_dict['is_normal'] = [p > 0.05 for p in jb_pvals]

    logger.info("estatísticas calculadas")
    return pd.DataFrame(stats_dict)
  

  def align_data(
      self,
      other_data: pd.DataFrame,
      method: str = "inner"
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if method == "inner":
      common_dates = self.returns.index.intersection(other_data.index)
      aligned_self = self.returns.loc[common_dates]
      aligned_other = other_data.loc[common_dates]

    elif method == "outer":
      all_dates = self.returns.index.union(other_data.index)
      aligned_self = self.returns.reindex(all_dates)
      aligned_other = other_data.reindex(all_dates)

    else:
      raise ValueError(f"metodo desconhecido: {method}")
    
    logger.info(f"data shape: {aligned_self.shape}")
    return aligned_self, aligned_other

  def save_processed_data(self, filename: str = "returns.parquet"):
    if self.returns is None:
      raise ValueError("nenhum retorno computado")
    
    filepath = PROCESSED_DATA_DIR / filename

    if filename.endswith('.parquet'):
      self.returns.to_parquet(filepath)
    elif filename.endswith('.csv'):
      self.returns.to_csv(filepath)
    else:
      raise ValueError(f"fomato de arquivo não suportado: {filename}")
    
    logger.info(f"data processada em {filepath}")

def separate_stocks_and_indices(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

  stock_cols = [col for col in prices.columns if '.SA' in col]
  index_cols = [col for col in prices.columns if col not in stock_cols]

  stocks = prices[stock_cols]
  indices = prices[index_cols]

  logger.info(f"separados: {len(stock_cols)} ações e {len(index_cols)} indices/benchmarks")
  return stocks, indices

def preprocess_stocks_separately(
  prices: pd.DataFrame,
  return_type: str = "log",
  handle_missing: str = "forward_fill",
  winsorize: bool = True,
  save: bool = True
) -> dict[str, any]:
  
  stocks, indices = separate_stocks_and_indices(prices)
  results = {}

  if len(stocks.columns) > 0:
    logger.info(f"\n[AÇÕES] preprocessando {len(stocks.columns)} ações")
    stock_preprocessor = DataPreprocessor(stocks)
    stock_preprocessor.handle_missing_data(method=handle_missing)
    stock_preprocessor.compute_returns(method=return_type)
  
    if winsorize:
      stock_preprocessor.returns = stock_preprocessor.winsorize_outliers(
      lower_percentile=0.01,
      upper_percentile=0.99
    )

    stocks_stationary = stock_preprocessor.check_stationary()
    stocks_stats = stock_preprocessor.get_descriptive_stats()

    results['stocks_returns'] = stock_preprocessor.returns
    results['stocks_stats'] = stocks_stats
    results['stocks_stationary'] = stocks_stationary

    if save:
      stock_preprocessor.save_processed_data("stocks_returns.parquet")
      stocks_stats.to_csv(PROCESSED_DATA_DIR / "stocks_descriptive_stats.csv")
      stocks_stationary.to_csv(PROCESSED_DATA_DIR / "stocks_stationary_tests.csv", index=False)
      logger.info("dados das ações salvos")

  if len(indices.columns) > 0:
    logger.info(f"\n[INDICES] Preprocessando {len(indices.columns)} indices e benchmarks")
    index_preprocessor = DataPreprocessor(indices)
    index_preprocessor.handle_missing_data(method=handle_missing)
    index_preprocessor.compute_returns(method=return_type)

    if winsorize:
      index_preprocessor.returns = index_preprocessor.winsorize_outliers(
        lower_percentile=0.01,
        upper_percentile=0.99
      )

    indices_stationary = index_preprocessor.check_stationary()
    indices_stats = index_preprocessor.get_descriptive_stats()

    results['indices_returns'] = index_preprocessor.returns
    results['indices_stats'] = indices_stats
    results['indices_stationary'] = indices_stationary

    if save:
      index_preprocessor.save_processed_data("indices_returns.parquet")
      indices_stats.to_csv(PROCESSED_DATA_DIR / "indices_stationarity_tests.csv", index=False)
      logger.info("dados dos indices salvos")

  if save and len(stocks.columns) > 0 and len(indices.columns) > 0:
    with open(PROCESSED_DATA_DIR / "comparative_summary.txt", 'w', encoding='utf-8') as f:
      f.write(f"período: {prices.index.min()} até {prices.index.max()}\n")

      f.write("ações brasileiras")
      f.write("quantidade: {len(stocks.columns)}")
      f.write(f"retorno médio anualizado: {stocks_stats['annual_return'].mean()*100:.2f}%")
      f.write(f"volatilidade média anualizada: {stocks_stats['annual_volatility'].mean()*100:.2f}%")
      f.write(f"sharpe ratio médio: {stocks_stats['sharpe_ratio'].mean():.3f}")
      f.write(f"séries estacionarias: {stocks_stationary['is_stationary'].sum()}/{len(stocks_stationary)}")

      f.write('top 5 ações por retorno')
      top_stocks = stocks_stats['annual_return'].sort_values(ascending=False).head(5)
      for ticker, ret in top_stocks.items():
        f.write(f"{ticker:15} {ret*100:7.2f}%")

      f.write("top 5 ações por sharpe ratio")
      top_sharpe_stocks = stocks_stats['sharpe_ratio'].sort_values(ascending=False).head(5)
      for ticker, sr in top_sharpe_stocks.items():
        f.write(f"{ticker:15} {sr:7.3f}")

      f.write("indices e benchmarks")
      f.write(f"quantidade: {len(indices.columns)}\n")
      f.write(f"retorno médio anualizado: {indices_stats['annual_return'].mean()*100:.2f}%\n")
      f.write(f"volatilidade média anualizada: {indices_stats['annual_volatility'].mean()*100:.2f}%\n")
      f.write(f"sharpe Ratio médio: {indices_stats['sharpe_ratio'].mean():.3f}\n")
      f.write(f"séries estacionárias: {indices_stationary['is_stationary'].sum()}/{len(indices_stationary)}\n\n")
            
      f.write("Performance dos índices:\n")
      for ticker in indices_stats.index:
        ret = indices_stats.loc[ticker, 'annual_return']
        vol = indices_stats.loc[ticker, 'annual_volatility']
        sr = indices_stats.loc[ticker, 'sharpe_ratio']
        f.write(f"  {ticker:15} | Retorno: {ret*100:7.2f}% | Vol: {vol*100:6.2f}% | Sharpe: {sr:6.3f}\n")
  
    logger.info("resumo salvo")
  
  logger.info("preprocessamento concluído")

  return results

def calculate_realized_volatility(
    returns: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
      logger.info(f"calculando volatilidade realizada (janela={window})")
      return returns.rolling(window=window).std()

def calculate_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.Series:
    logger.info(f"calculando correlação rolling (janela={window})")
    rolling_corr = returns.rolling(window=window).corr()
    avg_corr = []
  
    for date in returns.index[window:]:
      try:
        corr_matrix = rolling_corr.loc[date]
        upper_tri = corr_matrix.where(
          np.triu(np.ones(corr_matrix.sharpe), k=1).astype(bool)
        )
        avg_corr.append(upper_tri.stack().mean())
      except:
        avg_corr.append(np.nan)

    result = pd.Series(avg_corr, index=returns.index[window:], name='avg_correlation')
    return result

if __name__ == "__main__":
  from data_loader import DataLoader

  loader = DataLoader()

  try:
    from utils.config import PROCESSED_DATA_DIR
    loader.data_dir = PROCESSED_DATA_DIR
    loader.load_data("portfolio_raw.parquet")
    print("dados carregados do arquivo parquet")
  except FileNotFoundError:
    print("arquivo parquet não encontrado, carregando csvs da pasta raw")
    from utils.config import RAW_DATA_DIR
    loader.data_dir = RAW_DATA_DIR
    loader.load_data()
    print("dados csv carregados com sucesso")


  prices = loader.get_adjusted_close()
  print(f"shape dos preços: {prices.shape}")
  print(f"período: {prices.index.min()} até {prices.index.max()}")
  print(f"ativos: {list(prices.columns)}")


  results = preprocess_stocks_separately(
    prices,
    return_type="log",
    handle_missing="forward_fill",
    winsorize=True,
    save=True
  )

  if 'stocks_stats' in results:
    print("estatísticas das ações:")
    stocks_stats = results['stocks_stats']
    print(stocks_stats[['mean', 'std', 'annual_return', 'annual_volatility', 'sharpe_ratio']].round(4))

  if 'indices_stats' in results:
    print("estatisticas dos indices")
    indices_stats = results['indices_stats']
    print(indices_stats[['mean', 'std', 'annual_return', 'annual_volatility', 'sharpe_ratio']].round(4))

  print("preprocessamento concluído")


  
    

    



