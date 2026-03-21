import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = BASE_DIR / 'data' / 'external'
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

BCB_API_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"

SERIES = {

'selic': {
  'code': 11,
  'name': 'Taxa Selic',
  'filename': 'selic_daily.csv'
},

'ipca': {
  'code': 433,
  'name': 'IPCA',
  'filename': 'ipca_monthly.csv'
}
}

def download_bcb_series(series_code: int, start_date:str, end_date: str) -> pd.DataFrame:

  start_formatted = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d/%m/%Y')
  end_formatted = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d/%m/%Y')

  url = f"{BCB_API_BASE}.{series_code}/dados"
  params = {
    'formato': 'json',
    'dataInicial': start_formatted,
    'dataFinal': end_formatted
  }

  logger.info(f"Baixando serie {series_code} de {start_date} até {end_date}")

  try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if not data:
      logger.warning(f"série {series_code} retornou vazia")
      return pd.DataFrame()
    
    df = pd.DataFrame(data)

    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df = df.rename(columns={'data': 'Date', 'valor': 'Rate'})

    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')

    df = df.dropna()

    df = df.sort_values('Date').reset_index(drop=True)

    logger.info(f"baixando {len(df)} registrados")

    return df
  except requests.exceptions.RequestException as e:
    logger.error(f"erro ao baixar série {series_code}: {str(e)}")
    return pd.DataFrame()

def process_selic_cdi(df: pd.DataFrame) -> pd.DataFrame:

  if df.empty:
    return df


  df['Rate_Daily_pct'] = df['Rate']

  df['Rate_Daily'] = df['Rate'] / 100

  df['Rate_Annual'] = (1 + df['Rate_Daily']) ** 252 - 1
  df['Rate_Annual_pct'] = df['Rate_Annual'] * 100

  df['Daily_Return'] = df['Rate_Daily']
  
  logger.info(f"processado: taxa média diária = {df['Rate_Daily_pct'].mean():.4f}%")
  logger.info(f"processado: taxa média anual = {df['Rate_Annual_pct'].mean():2f}%")

  return df

def create_consolidated_risk_free_rate():

  logger.info("cruiando arquivo de taxa livre de risco")

  selic_path = EXTERNAL_DIR / 'selic_daily.csv'
  cdi_path = EXTERNAL_DIR / 'cdi_daily.csv'

  if not selic_path.exists() or not cdi_path.exists():
    logger.warning("execute download primeiro")
    return
  
  selic = pd.read_csv(selic_path, parse_dates=['Date'])
  cdi = pd.read_csv(cdi_path, parse_dates=['Date'])

  merged = pd.merge(
    selic[['Date', 'Rate_Daily']],
    cdi[['Date', 'Rate_Daily']],
    on='Date',
    how='outer',
    suffixes=('_Selic', '_CDI')
  )

  merged = merged.sort_values('Date').reset_index(drop=True)

  merged = merged.ffill()

  merged['Risk_Free_Rate'] = merged['Rate_Daily_Selic'].fillna(merged['Rate_Daily_CDI'])

  output_path = EXTERNAL_DIR / 'risk_free_rate.csv'
  merged[['Date', 'Risk_Free_Rate', 'Rate_Daily_Selic', 'Rate_Daily_CDI']].to_csv(
    output_path, index=False
  )

  logger.info(f"arquivo consolidado salvo: {output_path}")
  logger.info(f"período: {merged['Date'].min()} até {merged['Date'].max()}")
  logger.info(f"{len(merged)} dias úteis")

  return merged

def download_all_risk_free_data():
  logger.info("download de taxas livres de risco")

  logger.info(f"período: {START_DATE} até {END_DATE}")
  logger.info(f"destino: {EXTERNAL_DIR}")

  for key, info in SERIES.items():
    logger.info(f"[{key.upper()}] {info['name']} (série {info['code']})")

    df = download_bcb_series(info['code'], START_DATE, END_DATE)

    if df.empty:
      logger.warning(f"série {key} vazia, pulando")
      continue

    if key in ['selic', 'cdi']:
      df = process_selic_cdi(df)

    output_path = EXTERNAL_DIR / info['filename']
    df.to_csv(output_path, index=False)

    logger.info(f"salvo: {output_path}")
    logger.info(f"registros: {len(df)}")
    logger.info(f"período {df['Date'].min()} até {df['Date'].max()}")

    print("preview")
    print(df.head())

  create_consolidated_risk_free_rate()
  
  logger.info("download concluído")

def load_risk_free_rate(
    start_date: str = None,
    end_date: str = None,
    frequency: str = 'daily'
) -> pd.DataFrame:
  
  filepath = EXTERNAL_DIR / 'risk_free_rate.csv'

  if not filepath.exists():
    logger.error(f"arquico não encontrado: {filepath}")
    logger.error(f"execute: python scripts/download_risk_free_rate.py")
    return pd.DataFrame()
  
  df = pd.read_csv(filepath, parse_dates=['Date'])

  if start_date:
    df = df[df['Date'] >= start_date]
  if end_date:
    df = df[df['Date'] <= end_date]

  if frequency == 'monthly':
    df = df.set_index('Date').resample('M').last().reset_index()
  elif frequency == 'annual':
    df = df.set_index('Date').resample('Y').last().reset_index()

  return df

def get_current_selic() -> float:
  df = load_risk_free_rate()

  if df.empty:
    logger.warning("dados não disponíveis, usando taxa padrão 10.75%")
    return 0.1075
  
  latest_daily = df['Risk_Free_Rate'].iloc[-1]
  annual_rate = (1 + latest_daily) ** 252 - 1

  logger.info(f"selic anual: {annual_rate*100:.2f}% a.a")

  return annual_rate

if __name__ == "__main__":

  download_all_risk_free_data()

  rf = load_risk_free_rate(start_date='2023-01-01')
  print(f"dados carregados: {len(rf)} registros")
  print(rf.tail())

  print(f"taxa Selic atual para ser usada nos modelos:")
  current = get_current_selic()
  print(f"risk_free_rate = {current:.6f} # {current*100:.2f}% a.a.")