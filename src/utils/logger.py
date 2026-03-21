import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class LogColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class ColoredFormatter(logging.Formatter):
  
  FORMATS = {
    logging.DEBUG: LogColors.OKCYAN + "%(levelname)-8s" + LogColors.ENDC + "%(name)s - %(message)s",
    logging.INFO: LogColors.OKGREEN + "%(levelname)-8s" + LogColors.ENDC + "%(name)s - %(message)s",
    logging.WARNING: LogColors.WARNING + "%(levelname)-8s" + LogColors.ENDC + "%(names)s - %(message)s",
    logging.ERROR: LogColors.FAIL + LogColors.BOLD + "%(levelname)-8s" + LogColors.ENDC + "%(name)s - %(message)s",
    logging.CRITICAL: LogColors.FAIL + LogColors.BOLD + "%(levelname)-8s" + LogColors.ENDC + "%(name)s - %(message)s",
      }
  
  def format(self, record):
    log_fmt = self.FORMATS.get(record.levelno)
    formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    return formatter.format(record)
  
def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_mode: str = 'a',
) -> logging.Logger:
  
  logger = logging.getLogger(name)

  if logger.handlers:
    return logger
  
  logger.setLevel(level)
  logger.propagate = False

  file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)-8s - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )

  if log_file is None:
    log_file = f"{name.replace('.', '_')}.log"

  log_path = LOGS_DIR / log_file

  file_handler = RotatingFileHandler(
    log_path,
    mode=file_mode,
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
  )  

  file_handler.setLevel(level)
  file_handler.setFormatter(file_formatter)
  logger.addHandler(file_handler)

  if console:   
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

  return logger

def get_logger(name:str) -> logging.Logger:
  return setup_logger(name, level=logging.INFO)

class LoggerContext:
  def __init__(self, name: str, level: int = logging.DEBUG):
    self.name = name
    self.level = level
    self.logger = logging.getLogger(name)
    self.old_level = self.logger.level

  def __enter__(self):
    self.logger.setLevel(self.level)
    return self.logger
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.logger.setLevel(self.old_level)

def log_execution_time(func):
  import time
  from functools import wraps

  @wraps(func)
  def wrapper(*args, **kwargs):
    logger = logging.getLogger(func.__module__)

    start_time = time.time()
    logger.info(f"iniciando {func.__name__}")

    try: 
      result = func(*args, **kwargs)
      elapsed = time.time() - start_time
      logger.info(f"{func.__name__} concluído em {elapsed:.2f}s")
      return result
    
    except Exception as e:
      elapsed = time.time() - start_time
      logger.error(f"{func.__name__} falhou após {elapsed:.2f}s: {str(e)}")
      raise

  return wrapper

def log_function_call(func):

  from functools import wraps

  @wraps(func)
  def wrapper(*args, **kwargs):
    logger = logging.getLogger(func.__module__)

    args_repr = [repr(a) for a in args]
    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
    signature = ", ".join(args_repr + kwargs_repr)

    logger.debug(f"chamando {func.__name}({signature})")
    result = func(*args, **kwargs)
    logger.debug(f"{func.__name__} retornou {result!r}")

    return result
  
  return wrapper

def clear_old_logs(days: int = 30):
  import time

  logger = get_logger(__name__)
  cutoff = time.time() - (days * 86400)

  removed = 0
  for log_file in LOGS_DIR.glob('*.log*'):
    if log_file.stat().st_mtime < cutoff:
      log_file.unlink()
      removed += 1
      logger.info(f"removido log antigo: {log_file.name}")

  
  if removed > 0:
    logger.info(f"limpeza concluída: {removed} logs removidos")
  else:
    logger.info("nenhum log antigo para remover")

project_logger = setup_logger(
  'dynamic_copula_evt',
  level=logging.INFO,
  log_file='main.log'
)


if __name__ == "__main__":

  logger = get_logger(__name__)

  logger.debug("mensagem de debug")
  logger.info("mensagem de info")
  logger.warning("mensagem de warning")
  logger.error("mensagem de erro")

  @log_execution_time
  def funcao_teste():
    import time
    time.sleep(1)
    return "OK"
  
  resultado = funcao_teste()

  print(f"logs salvos em {LOGS_DIR}")
  print(f"arquivos de log: {list(LOGS_DIR.glob('*.log'))}")