from pathlib import Path

# ações brasileiras

ACOES_BRASILEIRAS = {
  'Setor Financeiro': {
    'ITUB4.SA': 'Itaú',
    'BBAS3.SA': 'Banco do Brasil',
    'BBDC4.SA': 'Bradesco',
    'B3SA3.SA': 'B3',
    'SANB11.SA': 'Santander Brasil'
  },
  'Commodities e Energia': {
    'PETR4.SA':'Petrobras',
    'VALE3.SA':'Vale',
    'SUZB3.SA': 'Suzano',
    'GGBR4.SA':'Gerdau'
  },
  'Utilities': {
    'ELET3.SA': 'Eletrobras',
    'TAEE11.SA': 'Taesa',
    'CMIG4.SA': 'Cemig',
    'EGIE3.SA': 'Engie'
  },
  'Consumo':{
    'ABEV3.SA':'Ambev',
    'RADL3.SA':'Raia Drogasil',
    'LREN3.SA':'Renner',
    'PCAR3.SA':'Pão de Açúcar',
  },
  'Industriais':{
    'WEGE3.SA':'WEG',
    'RENT3.SA':'localiza',
    'RAIL3.SA':'Rumo Logística'
  }
}

# Ativos internacionais e benchmarks
ATIVOS_INTERNACIONAIS = {
  'SPY': 'S&P 500 ETF',
  'EWZ': 'iShares Brazil ETF',
  'GLD': 'Gold ETF',
  'TLT': 'US Treasury 20Y',
  'UUP': 'US Dollar Index'
}

# índices de mercado
INDICES = {
  '^BVSP': 'Ibovespa',
  '^VIX': 'Volatility Index',
  'BRL=X': 'USD/BRL'
}

# definindo os diretorios base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# criando os diretorios caso não existam
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR]:
  directory.mkdir(parents=True, exist_ok=True)

PATHS = {
  'raw_data': str(RAW_DATA_DIR),
  'processed_data': str(PROCESSED_DATA_DIR),
  'reports': str(REPORTS_DIR),
  'figures': str(FIGURES_DIR)
}

# organizando os tickers

def get_all_tickers():
  tickers = [] # retornando em uma lista apenas os tickers

  for setor in ACOES_BRASILEIRAS.values():
    tickers.extend(setor.keys())
  tickers.extend(ATIVOS_INTERNACIONAIS.keys())
  tickers.extend(INDICES.keys())
  return tickers

def get_portfolio_tickers():
  tickers = []
  for setor in ACOES_BRASILEIRAS.values():
    tickers.extend(setor.keys())
  return tickers

# retorna os tickers de benchmark e indices
def get_benchmark_tickers():
  tickers = []
  tickers.extend(ATIVOS_INTERNACIONAIS.keys())
  tickers.extend(INDICES.keys())
  return tickers

#retornando os tickers com um nome mais limpo
def get_ticker_name(ticker):
  for setor in ACOES_BRASILEIRAS.values():
    if ticker in setor:
      return setor[ticker]
    
  if ticker in ATIVOS_INTERNACIONAIS:
      return ATIVOS_INTERNACIONAIS[ticker]
    
  if ticker in INDICES:
      return INDICES[ticker]
    
  return ticker
  

# configurações de download
DATA_CONFIG = {
  'start_date': '2020-01-01',
  'end_date': None,
  'interval': '1d',
  'auto_adjust': True,
  'threads': True
}
  
if __name__ == "__main__":
  print("configuração de ativos")

  print(f"total de ativos: {len(get_all_tickers())}")
  print(f"Ações brasileiras: {len(get_portfolio_tickers())}")
  print(f"Benchmarks e índives: {len(get_benchmark_tickers())}")

  print("ações brasileiras por setor")
  for setor, acoes in ACOES_BRASILEIRAS.items():
    print(f"{setor}:")
    for ticker, nome in acoes.items():
      print(f"{ticker:12} - {nome}")

  print("ativos internacionais e benchmarks")
  for ticker, nome in ATIVOS_INTERNACIONAIS.items():
    print(f"{ticker:12} - {nome}")

  print("indices")
  for ticker, nome in INDICES.items():
    print(f"{ticker:12} - {nome}")