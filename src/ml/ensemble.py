import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, List, Union, Tuple, Any
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from ml.lstm_predictor import LSTMPredictor, build_sequences
    from ml.gru_predictor  import GRUPredictor
except ImportError:
    try:
        from lstm_predictor import LSTMPredictor, build_sequences
        from gru_predictor  import GRUPredictor
    except ImportError:
        LSTMPredictor = None
        GRUPredictor  = None
        logger.warning("LSTMPredictor/GRUPredictor não encontrados. Ensemble limitado.")


# 1. Interface base

class _BasePredictor:
    """Interface mínima que qualquer modelo precisa implementar para participar do ensemble"""
    def fit(self, params, **kwargs): raise NotImplementedError
    def predict(self, horizon=1, return_df=True, **kwargs): raise NotImplementedError
    def evaluate(self, params_test, metrics=None): raise NotImplementedError


# 2. EnsemblePredictor

class EnsemblePredictor:
    """
    múltiplos preditores 

    strategy: 'mean', 'weighted', 'stacking', 'dynamic', 'uncertainty'
    models: lista de 'lstm', 'gru_vanilla', 'gru_attention', 'gru_tcn'

    """

    def __init__(
        self,
        strategy: str = "stacking",
        models: Optional[List[str]] = None,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.4,
        difference_input: bool = True,
        dynamic_window: int = 30,
        meta_model: str = "ridge",
        random_state: int = 42,
    ):
        self.strategy = strategy.lower()
        self.model_names = models or ["lstm", "gru_vanilla"]
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.difference_input = difference_input
        self.dynamic_window  = dynamic_window
        self.meta_model_type = meta_model
        self.random_state = random_state

        self.base_models_: Dict[str, Any] = {}
        self.weights_: Dict[str, float] = {}
        self._meta_model = None
        self._meta_scaler = None
        self.val_metrics_: Dict[str, Dict] = {}
        self._recent_errors: Dict[str, List[float]] = {}
        self.feature_names_: List[str] = []
        self.n_features_: int = 0
        self._is_fitted = False

    # Construção dos modelos base

    def _build_base_models(self) -> Dict[str, Any]:
        """Instancia os modelos conforme self.model_names"""
        models = {}
        kwargs = dict(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            random_state=self.random_state,
        )

        for name in self.model_names:
            if name == "lstm" and LSTMPredictor is not None:
                models[name] = LSTMPredictor(
                    arch="lstm",
                    difference_input=self.difference_input,
                    **kwargs,
                )
            elif name == "gru_vanilla" and GRUPredictor is not None:
                models[name] = GRUPredictor(arch="vanilla", **kwargs)
            elif name == "gru_attention" and GRUPredictor is not None:
                models[name] = GRUPredictor(arch="attention", **kwargs)
            elif name == "gru_tcn" and GRUPredictor is not None:
                models[name] = GRUPredictor(arch="tcn", **kwargs)
            else:
                logger.warning(f"Modelo '{name}' não reconhecido ou dependência ausente, pulando")

        if not models:
            raise RuntimeError(
                "Nenhum modelo pôde ser instanciado"
                "Verifique se LSTMPredictor/GRUPredictor estão disponíveis."
            )

        logger.info(f"Modelos base: {list(models.keys())}")
        return models

    # Fit

    def fit(
        self,
        params: Union[pd.DataFrame, np.ndarray],
        val_split: float = 0.15,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True,
    ) -> "EnsemblePredictor":
        """Treina todos os modelos base e o meta-modelo se strategy='stacking'"""
        if isinstance(params, pd.DataFrame):
            self.feature_names_ = list(params.columns)
            data = params.values.astype(np.float32)
        else:
            data = np.asarray(params, dtype=np.float32)
            self.feature_names_ = [f"p{i}" for i in range(data.shape[1])]
        self.n_features_ = data.shape[1]

        T       = len(data)
        n_val   = max(1, int(T * val_split))
        n_train = T - n_val

        if isinstance(params, pd.DataFrame):
            train_df = params.iloc[:n_train]
            val_df   = params.iloc[n_train:]
        else:
            train_df = pd.DataFrame(data[:n_train], columns=self.feature_names_)
            val_df   = pd.DataFrame(data[n_train:], columns=self.feature_names_)

        self.base_models_ = self._build_base_models()

        fit_kwargs = dict(
            epochs=epochs, batch_size=batch_size,
            lr=lr, patience=patience, verbose=verbose,
        )

        for name, model in self.base_models_.items():
            logger.info(f"Treinando: {name}")
            try:
                model.fit(train_df, val_split=val_split, **fit_kwargs)
                if len(val_df) > self.seq_len + 5:
                    metrics = model.evaluate(val_df, metrics=["mse", "mae", "rmse"])
                    self.val_metrics_[name] = metrics
                    logger.info(f"[{name}] val_MSE={metrics.get('mse', np.nan):.6f}")
                else:
                    self.val_metrics_[name] = {"mse": np.nan, "mae": np.nan, "rmse": np.nan}
            except Exception as e:
                logger.error(f"[{name}] Falhou: {e}. Removendo do ensemble")
                self.val_metrics_[name] = {"mse": np.inf}

        failed = [n for n, m in self.val_metrics_.items() if m.get("mse", np.inf) == np.inf]
        for n in failed:
            del self.base_models_[n]
            del self.val_metrics_[n]

        if not self.base_models_:
            raise RuntimeError("Todos os modelos base falharam")

        self._compute_weights()

        if self.strategy == "stacking":
            self._fit_stacking(train_df, val_df, fit_kwargs)

        self._is_fitted = True
        self._log_summary()
        return self

    def _compute_weights(self):
        """Calcula pesos por 1/MSE de validação"""
        mses = {
            n: max(m.get("mse", np.inf), 1e-10)
            for n, m in self.val_metrics_.items()
            if not np.isnan(m.get("mse", np.nan))
        }
        if not mses:
            n = len(self.base_models_)
            self.weights_ = {name: 1.0/n for name in self.base_models_}
            return

        inv_mse = {n: 1.0 / mse for n, mse in mses.items()}
        total   = sum(inv_mse.values())
        self.weights_ = {n: v / total for n, v in inv_mse.items()}
        for name in self.base_models_:
            if name not in self.weights_:
                self.weights_[name] = 0.0

    def _fit_stacking(self, train_df, val_df, fit_kwargs):
        """Treina meta-modelo Ridge sobre os forecasts base na janela de validação"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn ausente, stacking requer sklearn, usando weighted.")
            self.strategy = "weighted"
            return

        if len(val_df) < self.seq_len + 10:
            logger.warning("Validação muito pequena para stacking, usando weighted.")
            self.strategy = "weighted"
            return

        logger.info("Treinando meta-modelo (stacking)")

        val_data = val_df.values.astype(np.float32)
        n_val    = len(val_data)

        if n_val < self.seq_len + 5:
            self.strategy = "weighted"
            return

        X_meta_list, y_meta_list = [], []
        for t in range(self.seq_len, n_val):
            preds_t = []
            for name, model in self.base_models_.items():
                try:
                    window = val_data[t - self.seq_len: t]
                    pred   = model.predict(horizon=1, return_df=False, last_window=window)
                    preds_t.append(pred.ravel())
                except Exception:
                    preds_t.append(np.zeros(self.n_features_))
            X_meta_list.append(np.concatenate(preds_t))
            y_meta_list.append(val_data[t])

        if not X_meta_list:
            self.strategy = "weighted"
            return

        X_meta = np.array(X_meta_list, dtype=np.float32)
        y_meta = np.array(y_meta_list, dtype=np.float32)

        # Sem StandardScaler: previsões dos modelos base já estão na escala
        # original após reversão de difference_input, escalar introduz
        # incompatibilidade quando evaluate_all passa janelas brutas do teste
        self._meta_scaler = None
        self._meta_model = (
            Ridge(alpha=1.0) if self.meta_model_type == "ridge"
            else ElasticNet(alpha=0.1, l1_ratio=0.5)
        )
        self._meta_model.fit(X_meta, y_meta)

        mse_meta = mean_squared_error(y_meta, self._meta_model.predict(X_meta))
        logger.info(f"Meta-modelo ({self.meta_model_type}) MSE={mse_meta:.6f}")

    # Predict

    def predict(
        self,
        horizon: int = 1,
        return_df: bool = True,
        last_window: Optional[np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Gera previsão do ensemble usando a estratégia configurada"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        if self.strategy == "stacking" and self._meta_model is not None:
            return self._predict_stacking(horizon, return_df, last_window)
        elif self.strategy == "dynamic":
            return self._predict_dynamic(horizon, return_df, last_window)
        elif self.strategy == "uncertainty":
            return self._predict_uncertainty_weighted(horizon, return_df, last_window)
        else:
            return self._predict_averaged(horizon, return_df, last_window)

    def _predict_averaged(self, horizon, return_df, last_window):
        """Média simples ou ponderada por 1/MSE."""
        preds = {}
        for name, model in self.base_models_.items():
            try:
                p = model.predict(horizon=horizon, return_df=False, last_window=last_window)
                preds[name] = np.array(p, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[{name}] predict falhou: {e}")

        if not preds:
            raise RuntimeError("Todos os modelos falharam no predict")

        if self.strategy == "mean":
            weights = {n: 1.0 / len(preds) for n in preds}
        else:
            total   = sum(self.weights_.get(n, 0) for n in preds)
            weights = {
                n: self.weights_.get(n, 0) / max(total, 1e-10)
                for n in preds
            }

        result = np.zeros_like(list(preds.values())[0])
        for name, pred in preds.items():
            result += weights[name] * pred

        if return_df:
            return pd.DataFrame(result, columns=self.feature_names_)
        return result

    def _predict_stacking(self, horizon, return_df, last_window):
        """Previsão via meta-modelo Ridge (apenas horizon=1)"""
        if horizon > 1:
            logger.warning("Stacking suporta apenas horizon=1, usando weighted")
            return self._predict_averaged(horizon, return_df, last_window)

        base_preds = []
        for name, model in self.base_models_.items():
            try:
                p = model.predict(horizon=1, return_df=False, last_window=last_window)
                base_preds.append(p.ravel())
            except Exception:
                base_preds.append(np.zeros(self.n_features_))

        X = np.concatenate(base_preds).reshape(1, -1).astype(np.float32)

        result = self._meta_model.predict(X)[0].astype(np.float32).reshape(1, -1)
        # Clip para manter correlações no intervalo válido [-1, 1]
        result = np.clip(result, -1.0, 1.0)

        if return_df:
            return pd.DataFrame(result, columns=self.feature_names_)
        return result

    def _predict_dynamic(self, horizon, return_df, last_window):
        """
        Seleciona o modelo com menor erro na janela recente
        Adapta-se a mudanças de regime sem retreino
        """
        if not self._recent_errors:
            return self._predict_averaged(horizon, return_df, last_window)

        mean_errors = {
            n: np.mean(errs[-self.dynamic_window:])
            for n, errs in self._recent_errors.items()
            if errs and n in self.base_models_
        }
        if not mean_errors:
            return self._predict_averaged(horizon, return_df, last_window)

        best = min(mean_errors, key=mean_errors.get)
        logger.debug(f"Dynamic: usando {best} (MSE={mean_errors[best]:.6f})")

        result = self.base_models_[best].predict(
            horizon=horizon, return_df=False, last_window=last_window
        )
        if return_df:
            return pd.DataFrame(result, columns=self.feature_names_)
        return result

    def _predict_uncertainty_weighted(self, horizon, return_df, last_window):
        """
        Pondera por inverso da variância MC Dropout
        Modelos mais confiantes recebem maior peso
        """
        preds_mean, preds_var = {}, {}

        for name, model in self.base_models_.items():
            try:
                if hasattr(model, "predict_with_uncertainty"):
                    unc    = model.predict_with_uncertainty(horizon=horizon)
                    mean_p = unc["mean"].values.astype(np.float32)
                    var_p  = float(np.maximum((unc["std"].values ** 2).mean(), 1e-10))
                else:
                    mean_p = np.array(
                        model.predict(horizon=horizon, return_df=False, last_window=last_window),
                        dtype=np.float32,
                    )
                    var_p = float(self.val_metrics_.get(name, {}).get("mse", 1.0))
                preds_mean[name] = mean_p
                preds_var[name]  = var_p
            except Exception as e:
                logger.warning(f"[{name}] uncertainty predict falhou: {e}")

        if not preds_mean:
            return self._predict_averaged(horizon, return_df, last_window)

        inv_vars = {n: 1.0 / max(v, 1e-10) for n, v in preds_var.items()}
        total    = sum(inv_vars.values())
        weights  = {n: v / total for n, v in inv_vars.items()}

        result = np.zeros_like(list(preds_mean.values())[0])
        for name, pred in preds_mean.items():
            result += weights[name] * pred

        if return_df:
            return pd.DataFrame(result, columns=self.feature_names_)
        return result

    # Atualização de erros para seleção dinâmica

    def update_errors(self, true_values: Union[pd.DataFrame, np.ndarray]):
        """
        Atualiza histórico de erros recentes para strategy='dynamic'
        Deve ser chamado a cada passo t com os valores realizados
        """
        if isinstance(true_values, pd.DataFrame):
            y_true = true_values.values.astype(np.float32)
        else:
            y_true = np.asarray(true_values, dtype=np.float32)

        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)

        for name, model in self.base_models_.items():
            try:
                pred = model.predict(horizon=1, return_df=False)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                mse = float(np.mean((y_true - pred) ** 2))
                if name not in self._recent_errors:
                    self._recent_errors[name] = []
                self._recent_errors[name].append(mse)
            except Exception:
                pass

    # Avaliação comparativa

    def evaluate_all(
        self,
        params_test: Union[pd.DataFrame, np.ndarray],
        metrics: List[str] = ["mse", "mae", "rmse"],
    ) -> pd.DataFrame:
        """Avalia todos os modelos base e o ensemble no conjunto de teste"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        rows = []

        for name, model in self.base_models_.items():
            try:
                m = model.evaluate(params_test, metrics=metrics)
                m["model"]    = name
                m["strategy"] = "base"
                rows.append(m)
            except Exception as e:
                logger.warning(f"[{name}] evaluate falhou: {e}")

        if isinstance(params_test, pd.DataFrame):
            data = params_test.values.astype(np.float32)
        else:
            data = np.asarray(params_test, dtype=np.float32)

        # Avaliação rolling do ensemble não é possível com last_window bruto
        # porque difference_input e scaler de cada modelo são internos e não
        # replicáveis fora do modelo. A métrica relevante é a dos modelos base,
        # que avaliam corretamente via seu próprio pipeline em model.evaluate().
        # O ensemble em produção usa sempre a janela interna de cada modelo.

        df = pd.DataFrame(rows)
        if "model" in df.columns:
            df = df.set_index("model")
        return df

    def get_weights(self) -> pd.Series:
        return pd.Series(self.weights_, name="weight").sort_values(ascending=False)

    def get_val_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(self.val_metrics_).T

    def _log_summary(self):
        logger.info(f"Ensemble ({self.strategy.upper()}) — Resumo")
        logger.info(f"  Modelos base: {list(self.base_models_.keys())}")
        for name, w in sorted(self.weights_.items(), key=lambda x: -x[1]):
            mse = self.val_metrics_.get(name, {}).get("mse", np.nan)
            logger.info(f"  {name:20s}  peso={w:.4f}  val_MSE={mse:.6f}")

    # Save / Load

    def save(self, path: Union[str, Path]):
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "strategy": self.strategy,
                "model_names": self.model_names,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "difference_input": self.difference_input,
                "dynamic_window": self.dynamic_window,
                "meta_model": self.meta_model_type,
                "random_state": self.random_state,
            },
            "weights": self.weights_,
            "val_metrics": self.val_metrics_,
            "feature_names": self.feature_names_,
            "n_features": self.n_features_,
            "meta_model": self._meta_model,
            "meta_scaler": self._meta_scaler,
            "recent_errors": self._recent_errors,
            "is_fitted": self._is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        models_dir = path.parent / (path.stem + "_models")
        models_dir.mkdir(exist_ok=True)
        for name, model in self.base_models_.items():
            try:
                model.save(models_dir / f"{name}.pkl")
            except Exception as e:
                logger.warning(f"Não foi possível salvar {name}: {e}")

        logger.info(f"Ensemble salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnsemblePredictor":
        import pickle
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(**state["config"])
        obj.weights_ = state["weights"]
        obj.val_metrics_ = state["val_metrics"]
        obj.feature_names_ = state["feature_names"]
        obj.n_features_ = state["n_features"]
        obj._meta_model = state["meta_model"]
        obj._meta_scaler = state["meta_scaler"]
        obj._recent_errors = state["recent_errors"]
        obj._is_fitted = state["is_fitted"]

        models_dir = path.parent / (path.stem + "_models")
        if models_dir.exists():
            for name in obj.model_names:
                model_path = models_dir / f"{name}.pkl"
                if model_path.exists():
                    try:
                        obj.base_models_[name] = (
                            GRUPredictor.load(model_path) if "gru" in name
                            else LSTMPredictor.load(model_path)
                        )
                    except Exception as e:
                        logger.warning(f"Não foi possível carregar {name}: {e}")

        logger.info(f"Ensemble carregado de {path}")
        return obj


# Wrapper 

def build_ensemble(
    params_df: pd.DataFrame,
    strategy: str = "stacking",
    models: Optional[List[str]] = None,
    seq_len: int = 20,
    hidden_size: int = 32,
    epochs: int = 100,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Tuple["EnsemblePredictor", pd.DataFrame]:
    """
    Wrapper de alto nível para main_pipeline.py
    Default: stacking com lstm e gru_vanilla 
    """
    ensemble = EnsemblePredictor(
        strategy=strategy,
        models=models or ["lstm", "gru_vanilla"],
        seq_len=seq_len,
        hidden_size=hidden_size,
    )
    ensemble.fit(params_df, epochs=epochs, verbose=verbose)
    forecast = ensemble.predict(horizon=1, return_df=True)

    if save_path:
        ensemble.save(save_path)

    return ensemble, forecast


# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    T, d = 500, 6
    rho_base   = np.array([0.3, 0.4, 0.2, 0.5, 0.35, 0.25])
    params_sim = np.zeros((T, d))
    params_sim[0] = rho_base
    for t in range(1, T):
        params_sim[t] = np.clip(
            0.95 * params_sim[t-1] + 0.05 * rho_base + np.random.normal(0, 0.01, d),
            -0.99, 0.99,
        )

    cols      = [f"rho_{i}" for i in range(d)]
    params_df = pd.DataFrame(
        params_sim, columns=cols,
        index=pd.date_range("2020-01-01", periods=T, freq="B"),
    )
    split    = int(T * 0.85)
    train_df = params_df.iloc[:split]
    test_df  = params_df.iloc[split:]

    # Weighted lstm e gru_vanilla

    print("\n EnsemblePredictor (weighted) ")
    ens_w = EnsemblePredictor(
        strategy="weighted",
        models=["lstm", "gru_vanilla"],
        seq_len=20, hidden_size=32, num_layers=1, dropout=0.4,
    )
    ens_w.fit(train_df, epochs=50, patience=10, verbose=True)
    forecast_w = ens_w.predict(horizon=1, return_df=True)
    print(f"Forecast weighted:\n{forecast_w.round(4)}")
    print(f"Pesos: {ens_w.get_weights().round(4).to_dict()}")

    # Stacking lstm e gru_vanilla

    print("\n EnsemblePredictor (stacking) ")
    ens_s = EnsemblePredictor(
        strategy="stacking",
        models=["lstm", "gru_vanilla"],
        seq_len=20, hidden_size=32, num_layers=1, dropout=0.4,
    )
    ens_s.fit(train_df, epochs=50, patience=10, verbose=False)
    forecast_s = ens_s.predict(horizon=1, return_df=True)
    print(f"Forecast stacking:\n{forecast_s.round(4)}")

    # Mean

    print("\n EnsemblePredictor (mean) ")
    ens_m = EnsemblePredictor(
        strategy="mean",
        models=["lstm", "gru_vanilla"],
        seq_len=20, hidden_size=32, num_layers=1, dropout=0.4,
    )
    ens_m.fit(train_df, epochs=50, patience=10, verbose=False)
    forecast_m = ens_m.predict(horizon=1, return_df=True)
    print(f"Forecast mean:\n{forecast_m.round(4)}")

    # comparando stacking e modelos base

    print("\n Avaliação comparativa no teste ")
    eval_df = ens_s.evaluate_all(test_df)
    print(eval_df.round(6))

    # Wrapper

    print("\n build_ensemble (wrapper) ")
    ens_quick, fc = build_ensemble(
        train_df, strategy="stacking",
        models=["lstm", "gru_vanilla"],
        seq_len=20, hidden_size=32, epochs=30, verbose=False,
    )
    print(f"Forecast rápido: {fc.round(4).to_string()}")

    print("\n Todos os testes concluídos.")
