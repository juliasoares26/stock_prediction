import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import sys
import logging
import time
from typing import Optional, Dict, List

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# adicionando o diretorio raiz ao path para importar o arquivo config.py
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

try:
  from config import(
    get_all_tickers,
    get_portfolio_tickers,
    get_benchmark_tickers,
    get_ticker_name,
    DATA_CONFIG,
    PATHS
)
  print("configurações importadas de config.py")

except ImportError:
  try:
    from utils.config import(
      get_all_tickers,
      get_portfolio_tickers,
      get_benchmark_tickers,
      get_ticker_name,
      DATA_CONFIG,
      PATHS,
      RAW_DATA_DIR,
      PROCESSED_DATA_DIR
    )
    print(f"configurações importadas de utils.config")
  except ImportError:
    raise ImportError('não foi possível importar')
  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
  
def safe_get_ticker_name(ticker):
  try:
    return get_ticker_name(ticker)
  except:
    return ticker

def create_directories():
  for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
  print('diretórios criados')

def download_ticker_data(ticker, start_date=None, end_date=None, max_retries=3):
  start = start_date or DATA_CONFIG.get('start_date', '2020-01-01')
  end = end_date or datetime.now().strftime('%Y-%m-%d')

  for attempt in range(max_retries):
    try:
      ticker_name = safe_get_ticker_name(ticker)

      if attempt > 0:
        print(f"tentativa {attempt + 1}/{max_retries} - {ticker:12}", end=" ", flush=True)
      else:
        print(f"baixando {ticker:12} ({ticker_name:20})", end=" ", flush=True) 

      ticker_obj = yf.Ticker(ticker)
      data = ticker_obj.history(
        start = start,
        end = end,
        interval=DATA_CONFIG.get('interval', '1d'),
        auto_adjust=DATA_CONFIG.get('auto_adjust', True),
        timeout=30
      )

      if data.empty:
        print('sem dados')
        return None
      
      print(f"{len(data):4} registros")
      return data
    
    except Exception as e:
      if attempt < max_retries - 1:
        print('tentando novamente')
        time.sleep(2)
      else:
        print(f"erro: {str(e)[:50]}")
        return None
      
  return None
  
def download_multiple_tickers(tickers, start_date=None, end_date=None, delay=0.5):
  data_dict = {}
  failed = []

  print(f"baixando {len(tickers)} ativos")

  for i, ticker in enumerate(tickers, 1):
    print(f"[{i}/{len(tickers)}]", end="")
    df = download_ticker_data(ticker, start_date, end_date)

    if df is not None:
      data_dict[ticker] = df
    else:
      failed.append(ticker)

    if i < len(tickers):
      time.sleep(delay)

  print(f"download concluído: {len(data_dict)}/{len(tickers)} ativos baixados")
  if failed:
    print(f"falharam: {', '.join(failed)}")

  return data_dict

def save_data(data_dict, filename_prefix='dados'):

  if not data_dict:
    print(f"nenhum dado para salvar com prefixo '{filename_prefix}'")
    return
  
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  os.makedirs(PATHS['raw_data'], exist_ok=True)

  print(f"salvando {len(data_dict)} arquivos em {os.path.abspath(PATHS['raw_data'])}")

  saved_count = 0
  for ticker, df in data_dict.items():
    try:
      safe_ticker = ticker.replace('=', '_').replace('^', '').replace('.', '_')
      filename = f"{PATHS['raw_data']}/{filename_prefix}_{safe_ticker}_{timestamp}.csv"

      df.to_csv(filename)

      if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"{ticker:12} salvo ({file_size:,} bytes)")
        saved_count += 1
      else:
        print(f"{ticker:12} não foi criado")
    except Exception as e:
      print(f"erro ao salvar {ticker:12}: {str(e)}")

  summary_file = f"{PATHS['raw_data']}/summary_{filename_prefix}_{timestamp}.txt"

  try:
    with open(summary_file, 'w', encoding='utf-8') as f:
      f.write(f"download realizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(f"Total de ativos: {len(data_dict)}\n\n")
      f.write("Ativos baixados:\n")
      for ticker in data_dict.keys():
        ticker_name = safe_get_ticker_name(ticker)
        num_records = len(data_dict[ticker])
        f.write(f" - {ticker:15} {ticker_name:25} ({num_records} registros)\n")
  except Exception as e:
    print(f"erro ao salvar resumo: {str(e)}")

  print(f"total {saved_count}/{len(data_dict)} arquivos salvos")

  try:
    files_in_dir = os.listdir(PATHS['raw_data'])
    print(f"arquivos na pasta data/raw/: {len(files_in_dir)}")
  except Exception as e:
    print(f"não foi possível listar arquivos: {e}")

def save_consolidated_parquet(data_dict, filepath):
  if not data_dict:
    return
  
  close_prices = {}
  volumes = {}

  for ticker, df in data_dict.items():
    if 'Close' in df.columns:
      close_prices[ticker] = df['Close']
    if 'Volume' in df.columns:
      volumes[ticker] = df['Volume']

  if close_prices:
    df_prices = pd.DataFrame(close_prices)

    if df_prices.index.tz is not None:
      df_prices.index = df_prices.index.tz_localize(None)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    df_prices.to_parquet(filepath, compression='snappy')
    logger.info(f"arquivo parquet salvo: {filepath} - shape: {df_prices.shape}")

    if volumes:
      df_volumes = pd.DataFrame(volumes)
      if df_volumes.index.tz is not None:
        df_volumes.index = df_volumes.index.tz_localize(None)

      volume_path = filepath.replace('.parquet', '_volumes.parquet')
      df_volumes.to_parquet(volume_path, compression='snappy')
      logger.info(f"Volumes salvos: {volume_path}")
      
def download_all_data(start_date=None, end_date=None, save=True):
  create_directories()

  print("baixando ações do portfólio")
  portfolio_tickers = get_portfolio_tickers()
  portfolio_data = download_multiple_tickers(portfolio_tickers, start_date, end_date)

  print("baixando benchmarks e índices")
  benchmark_tickers = get_benchmark_tickers()
  benchmark_data = download_multiple_tickers(benchmark_tickers, start_date, end_date)

  if save:
    print("salvando dados")
    save_data(portfolio_data, 'portfolio')
    save_data(benchmark_data, 'benchmark')

  print(f"ações do portfólio: {len(portfolio_data)}/{len(portfolio_tickers)}")
  print(f"benchmarks e índices: {len(benchmark_data)}/{len(benchmark_tickers)}")
  print(f"total de ativos: {len(portfolio_data) + len(benchmark_data)}")

  if save:
    print("salvando dados consolidados em parquet")
    save_consolidated_parquet(portfolio_data, 'data/raw/b3_stocks_daily.parquet')
    save_consolidated_parquet(benchmark_data, 'data/raw/market_indices.parquet')
  return portfolio_data, benchmark_data
 

class DataLoader:
  def __init__(self, data_dir: Optional[Path] = None):
    try:
      self.data_dir = data_dir or RAW_DATA_DIR
    except:
      self.data_dir = data_dir or Path(PATHS['raw_data'])

    self.data = {}
    self.prices = None
    self.volumes = None

  def load_data(
      self,
      filename: Optional[str] = None,
      tickers: Optional[List[str]] = None,
      start_date: Optional[str] = None,
      end_date: Optional[str] = None
  ) -> Dict[str, pd.DataFrame]:
    
    if filename:
      filepath = self.data_dir / filename
      if not filepath.exists():
        raise FileNotFoundError(f"arquivo não encontrado: {filepath}" )
    
      logger.info(f"carregando dados de {filepath}")

      if filename.endswith('.parquet'):
        df = pd.read_parquet(filepath)
        if isinstance(df, pd.DataFrame):
          self.prices = df
          for col in df.columns:
            self.data[col] = pd.DataFrame({'Close': df[col]})

      elif filename.endswith('.csv'):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.data = {'all': df}

      else:
        raise ValueError(f"Formato não suportado: {filename}")
      
    else:
      self.data = self._load_csv_files(tickers)

    if start_date or end_date:
      self._filter_by_date(start_date, end_date)

    self._organize_data()

    logger.info(f"dados carregados: {len(self.data)} tickers")
    return self.data
  
  def _load_csv_files(self, tickers: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    data_dict = {}

    if not self.data_dir.exists():
      logger.warning(f"Diretório não encontrado: {self.data_dir}")
      return data_dict
    
    csv_files = list(self.data_dir.glob('*.csv'))

    if not csv_files:
      logger.warning(f"nenhum arquivo CSV encontrado em {self.data_dir}")
      return data_dict
    
    target_tickers = tickers or get_all_tickers()

    ticker_files = {}
    for file in csv_files:
      if 'summary' in file.name.lower():
        continue

      parts = file.stem.split('_')
      if len(parts) < 3:
        continue

      ticker_parts = parts[1:-2]
      ticker = '_'.join(ticker_parts)

      if '_SA' in ticker:
        ticker = ticker.replace('_SA', '.SA')

      if ticker == 'BRL_X':
        ticker = 'BRL=X'

      if ticker == 'BVSP':
        ticker = '^BVSP'
      elif ticker == 'VIX':
        ticker = '^VIX'

      if ticker in target_tickers:
        if ticker not in ticker_files or file.stat().st_mtime > ticker_files[ticker].stat().st_mtime:
          ticker_files[ticker] = file

    for ticker, file in ticker_files.items():
      try:
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if not df.empty:
          data_dict[ticker] = df
          logger.debug(f"carregado {ticker}: {len(df)} registros")

        else:
          logger.warning(f"arquivo vazio: {file.name}")

      except Exception as e:
        logger.error(f"Erro ao carregar {file.name}: {str(e)}")
 
    logger.info(f"carregados {len(data_dict)} arquivos CSV")
    return data_dict
    

  def _filter_by_date(self, start_date: Optional[str], end_date: Optional[str]):
    for ticker in self.data:
      df = self.data[ticker]

      if start_date:
        df = df[df.index >= start_date]

      if end_date:
        df = df[df.index <= end_date]

      self.data[ticker] = df

  def _organize_data(self):

    if not self.data:
      return
    
    price_data = {}
    volume_data = {}

    for ticker, df in self.data.items():
        df_clean = df.copy()

        if isinstance(df_clean.index, pd.DatetimeIndex):
          if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_convert('UTC').tz_localize(None)

        else:
          try:
            df_clean.index = pd.to_datetime(df_clean.index, utc=True).tz_localize(None)
          except Exception as e:
            logger.warning(f"erro ao converter indice de {ticker}: {e}")
            df_clean.index = pd.to_datetime(df_clean.index, errors='coerce')

        if 'Close' in df_clean.columns:
          price_data[ticker] = df_clean['Close']
  
        if 'Volume' in df.columns:
          volume_data[ticker] = df_clean['Volume']
    
    if price_data:
      self.prices = pd.DataFrame(price_data)
      nan_before = self.prices.isnull().sum().sum()
      logger.info(f"Dataframe de preços criados: {self.prices.shape}")
      logger.info(f"NaN restantes: {self.prices.isnull().sum().sum()}")

      self.prices = self.prices.ffill().bfill()

      nan_after = self.prices.isnull().sum().sum()
      logger.info(f"NaN após limpezas: {nan_after}")

      if nan_after > 0:
        logger.warning(f"ainda há {nan_after} valores Nan. removendo linhas")
        self.prices = self.prices.dropna()
        logger.info(f"shape final: {self.prices.shape}")

    if volume_data:
      self.volumes = pd.DataFrame(volume_data)
      self.volumes = self.volumes.fillna(0)
      logger.info(f"DataFrame de volumes criado: {self.volumes.shape}")


  def get_adjusted_close(self) -> pd.DataFrame:
    if self.prices is None:
      raise ValueError("dados não carregados, execute load_data() primeiro")
    return self.prices
  
  def get_volumes(self) -> pd.DataFrame:
    if self.volumes is None:
      raise ValueError("dados não carregados, execute load_data() primeiro")
    return self.volumes
    
  def get_ticker_data(self, ticker: str) -> pd.DataFrame:
    if ticker not in self.data:
      raise ValueError(f"ticker {ticker} não encontrado nos dados carregados")
    return self.data[ticker]
  
  def save_consolidated(
      self,
      filename: str = 'portfolio_raw.parquet',
      format: str = 'parquet'
  ):
    
    if self.prices is None:
      raise ValueError('nenhum dado para salvar')
    
    try:
      filepath = PROCESSED_DATA_DIR / filename
    except:
      filepath = Path(PATHS['processed_data']) / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'parquet':
      self.prices.to_parquet(filepath)
    elif format == 'csv':
      self.prices.to_csv(filepath)
    else:
      raise ValueError(f"formato não suportado: {format}")
    
    logger.info(f"dados consolidados salvos em {filepath}")


  def get_data_summary(self) -> pd.DataFrame:
    summary = []


    for ticker in self.data:
      df = self.data[ticker]
      summary.append({
        'Ticker': ticker,
        'Nome': safe_get_ticker_name(ticker),
        'Registros': len(df),
        'Data Início': df.index.min(),
        'Data Fim': df.index.max(),
        'Dias': (df.index.max() - df.index.min()).days,
        'Missing': df.isnull().sum().sum()
      })

    return pd.DataFrame(summary)

def load_latest_data():
  return download_all_data()

if __name__ == "__main__":
  portfolio_data, benchmark_data = download_all_data()

  if portfolio_data:
    primeiro_ticker = list(portfolio_data.keys())[0]
    print(f"exemplo de dados ({primeiro_ticker}):")
    print(portfolio_data[primeiro_ticker].head())
    print(f"shape: {portfolio_data[primeiro_ticker].shape}")

  print("testando data loader")

  loader = DataLoader()
  loader.load_data()

  if loader.prices is not None:
    print(f"\n preços consolidados: {loader.prices.shape}")
    print(f"\n primeiras linhas")
    print(loader.prices.head())

    loader.save_consolidated()

    print("\n resumo dos dados:")
    print(loader.get_data_summary())