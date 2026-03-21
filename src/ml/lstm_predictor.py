import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch não disponível. Usando fallback sklearn (ARIMA-like via MLP)")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível")


# Construção de sequências

def build_sequences(
    data: np.ndarray,
    seq_len: int,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma array temporal em pares (X, y) para treino supervisionado
    """
    T, n_features = data.shape
    X_list, y_list = [], []

    for i in range(T - seq_len - horizon + 1):
        X_list.append(data[i : i + seq_len])
        y_window = data[i + seq_len : i + seq_len + horizon]
        if horizon == 1:
            y_list.append(y_window[0])
        else:
            y_list.append(y_window)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


# 2. Módulos de rede neural

if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        """
        Arquitetura LSTM multi-layer com dropout
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.4,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    class _GRUNet(nn.Module):
        """GRU como alternativa mais leve ao LSTM"""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.4,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)


# LSTMPredictor

class LSTMPredictor:
    """
    Previsão de parâmetros de cópulas usando LSTM ou GRU
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.4,
        difference_input: bool = True,
        arch: str = "lstm",
        scaler: str = "standard",
        device: str = "auto",
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.difference_input = difference_input
        self.arch = arch.lower()
        self.scaler_type = scaler
        self.random_state = random_state

        if TORCH_AVAILABLE:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = None

        self.model = None
        self.scaler = None
        self.feature_names_: List[str] = []
        self.n_features_: int = 0
        self.train_history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted = False
        self._fallback_model = None
        self._fallback_data = None

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch ausente — usando fallback Ridge Regression com features lag")

    # Pré-processamento

    def _init_scaler(self):
        if not SKLEARN_AVAILABLE:
            return None
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"scaler desconhecido: {self.scaler_type}")

    def _prepare_data(
        self,
        params: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Converte input para np.ndarray float32
        Aplica diferenciação de primeira ordem se difference_input=True para estacionarizar correlações rolling (autocorr ~0.93 -> ~0.49)
        """
        if isinstance(params, pd.Series):
            params = params.to_frame()

        if isinstance(params, pd.DataFrame):
            self.feature_names_ = list(params.columns)
            data = params.values.astype(np.float32)
        elif isinstance(params, np.ndarray):
            data = params.astype(np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self.feature_names_ = [f"param_{i}" for i in range(data.shape[1])]
        else:
            raise TypeError(f"tipo não suportado: {type(params)}")

        mask = np.any(np.isnan(data), axis=1)
        if mask.any():
            logger.warning(f"removendo {mask.sum()} linhas com NaN antes do treino")
            df_tmp = pd.DataFrame(data).ffill().bfill()
            data = df_tmp.values.astype(np.float32)

        self.n_features_ = data.shape[1]

        if self.difference_input:
            self._last_level = data[-1:].copy()
            data = np.diff(data, axis=0)

        self.n_features_ = data.shape[1]
        return data

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
    ) -> "LSTMPredictor":
        """
        Treina o modelo nos parâmetros históricos
        """
        np.random.seed(self.random_state)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)

        data = self._prepare_data(params)
        T = data.shape[0]

        if T < self.seq_len + 10:
            raise ValueError(
                f"Dados insuficientes: T={T}, seq_len={self.seq_len}. "
                f"Reduza seq_len para no máximo {T - 10}."
            )

        self.scaler = self._init_scaler()
        if self.scaler is not None:
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = data.copy()

        if not TORCH_AVAILABLE:
            return self._fit_fallback(data_scaled, val_split, verbose)

        X, y = build_sequences(data_scaled, self.seq_len, horizon=1)
        n_samples = X.shape[0]

        n_val = max(1, int(n_samples * val_split))
        n_train = n_samples - n_val

        X_train = torch.tensor(X[:n_train], dtype=torch.float32)
        y_train = torch.tensor(y[:n_train], dtype=torch.float32)
        X_val = torch.tensor(X[n_train:], dtype=torch.float32)
        y_val = torch.tensor(y[n_train:], dtype=torch.float32)

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        NetClass = _LSTMNet if self.arch == "lstm" else _GRUNet
        self.model = NetClass(
            input_size=self.n_features_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.n_features_,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience // 2, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val.to(self.device))
                val_loss = criterion(val_pred, y_val.to(self.device)).item()

            train_loss = np.mean(train_losses)
            self.train_history_["train_loss"].append(train_loss)
            self.train_history_["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                logger.info(
                    f"Epoch {epoch:4d}/{epochs}  "
                    f"train_loss={train_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  "
                    f"patience={patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping na época {epoch}. Melhor val_loss={best_val_loss:.6f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._data_scaled = data_scaled
        self._is_fitted = True
        logger.info(
            f"Treino concluído  arch={self.arch}  "
            f"features={self.n_features_}  device={self.device}"
        )
        return self

    # Fit fallback 

    def _fit_fallback(self, data_scaled, val_split, verbose):
        """Fallback com Ridge Regression para ambientes sem PyTorch."""
        X, y = build_sequences(data_scaled, self.seq_len, horizon=1)
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)

        n_val = max(1, int(n_samples * val_split))
        n_train = n_samples - n_val

        self._fallback_model = Ridge(alpha=1.0)
        self._fallback_model.fit(X_flat[:n_train], y[:n_train])

        if verbose:
            y_val_pred = self._fallback_model.predict(X_flat[n_train:])
            mse = mean_squared_error(y[n_train:], y_val_pred)
            logger.info(f"[Fallback Ridge] val MSE={mse:.6f}")

        self._data_scaled = data_scaled
        self._is_fitted = True
        return self

    # Predict

    def predict(
        self,
        horizon: int = 1,
        return_df: bool = True,
        last_window: Optional[np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Prevê os próximos 'horizon' passos dos parâmetros de cópula
        Reverte a diferenciação acumulando deltas sobre o último nível observado
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não treinado. Execute fit() primeiro")

        if last_window is None:
            window = self._data_scaled[-self.seq_len:].copy()
        else:
            if self.scaler is not None:
                window = self.scaler.transform(last_window)
            else:
                window = last_window.copy()

        predictions_scaled = []

        for _ in range(horizon):
            inp = window[-self.seq_len:]

            if TORCH_AVAILABLE and self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(
                        inp[np.newaxis, :, :], dtype=torch.float32
                    ).to(self.device)
                    pred = self.model(x_tensor).cpu().numpy()[0]
            elif self._fallback_model is not None:
                x_flat = inp.reshape(1, -1)
                pred = self._fallback_model.predict(x_flat)[0]
            else:
                raise RuntimeError("Nenhum modelo disponível para previsão")

            predictions_scaled.append(pred)
            window = np.vstack([window, pred])

        predictions_scaled = np.array(predictions_scaled, dtype=np.float32)

        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions_scaled)
        else:
            predictions = predictions_scaled

        if self.difference_input and hasattr(self, "_last_level"):
            last = self._last_level.copy()
            reverted = np.zeros_like(predictions)
            for h in range(predictions.shape[0]):
                last = last + predictions[h:h+1]
                reverted[h] = last
            predictions = np.clip(reverted, -1.0, 1.0)

        if return_df:
            return pd.DataFrame(predictions, columns=self.feature_names_)
        return predictions

    # Avaliação

    def evaluate(
        self,
        params_test: Union[pd.DataFrame, np.ndarray],
        metrics: List[str] = ["mse", "mae", "rmse"],
    ) -> Dict[str, float]:
        """
        Avalia o modelo em dados de teste
        """
        data = self._prepare_data(params_test)
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data.copy()

        X, y_true = build_sequences(data_scaled, self.seq_len, horizon=1)

        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                y_pred_scaled = self.model(x_tensor).cpu().numpy()
        elif self._fallback_model is not None:
            X_flat = X.reshape(len(X), -1)
            y_pred_scaled = self._fallback_model.predict(X_flat)
        else:
            raise RuntimeError("Modelo não treinado.")

        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred_scaled)
            y_true_inv = self.scaler.inverse_transform(y_true)
        else:
            y_pred = y_pred_scaled
            y_true_inv = y_true

        results = {}
        if "mse" in metrics:
            results["mse"] = float(np.mean((y_true_inv - y_pred) ** 2))
        if "mae" in metrics:
            results["mae"] = float(np.mean(np.abs(y_true_inv - y_pred)))
        if "rmse" in metrics:
            results["rmse"] = float(np.sqrt(np.mean((y_true_inv - y_pred) ** 2)))
        if "mape" in metrics:
            mask = y_true_inv != 0
            if mask.any():
                results["mape"] = float(
                    np.mean(np.abs((y_true_inv[mask] - y_pred[mask]) / y_true_inv[mask]))
                ) * 100
            else:
                results["mape"] = np.nan

        logger.info(f"Avaliação: {results}")
        return results

    # Save / Load

    def save(self, path: Union[str, Path]):
        """Salva modelo treinado"""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "difference_input": self.difference_input,
                "arch": self.arch,
                "scaler_type": self.scaler_type,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names_,
            "n_features": self.n_features_,
            "scaler": self.scaler,
            "train_history": self.train_history_,
            "data_scaled": self._data_scaled if hasattr(self, "_data_scaled") else None,
            "last_level": self._last_level if hasattr(self, "_last_level") else None,
            "is_fitted": self._is_fitted,
        }

        if TORCH_AVAILABLE and self.model is not None:
            state["model_state_dict"] = self.model.state_dict()
            state["backend"] = "torch"
        else:
            state["fallback_model"] = self._fallback_model
            state["backend"] = "sklearn"

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Modelo salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTMPredictor":
        """Carrega modelo salvo."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls(**state["config"])
        obj.feature_names_ = state["feature_names"]
        obj.n_features_ = state["n_features"]
        obj.scaler = state["scaler"]
        obj.train_history_ = state["train_history"]
        obj._is_fitted = state["is_fitted"]

        if state.get("data_scaled") is not None:
            obj._data_scaled = state["data_scaled"]
        if state.get("last_level") is not None:
            obj._last_level = state["last_level"]

        if state["backend"] == "torch" and TORCH_AVAILABLE:
            NetClass = _LSTMNet if obj.arch == "lstm" else _GRUNet
            obj.model = NetClass(
                input_size=obj.n_features_,
                hidden_size=obj.hidden_size,
                num_layers=obj.num_layers,
                output_size=obj.n_features_,
                dropout=obj.dropout,
            ).to(obj.device)
            obj.model.load_state_dict(state["model_state_dict"])
        else:
            obj._fallback_model = state.get("fallback_model")

        logger.info(f"Modelo carregado de {path}")
        return obj

    def get_training_history(self) -> pd.DataFrame:
        """Retorna histórico de treino como DataFrame."""
        return pd.DataFrame(self.train_history_)


#  Wrapper de alto nível

def fit_predict_copula_params(
    params_df: pd.DataFrame,
    seq_len: int = 20,
    horizon: int = 1,
    arch: str = "lstm",
    epochs: int = 100,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.4,
    difference_input: bool = True,
    val_split: float = 0.15,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple["LSTMPredictor", pd.DataFrame]:
    """
    Wrapper, treina LSTM nos parâmetros e retorna previsão
    """
    if params_df.isnull().all().all():
        raise ValueError("params_df está vazio ou todo NaN.")

    logger.info(
        f"Iniciando LSTM pipeline arch={arch}  "
        f"T={len(params_df)} params={len(params_df.columns)}  "
        f"seq_len={seq_len} horizon={horizon}"
    )

    predictor = LSTMPredictor(
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        difference_input=difference_input,
        arch=arch,
    )

    predictor.fit(params_df, epochs=epochs, val_split=val_split, verbose=verbose)
    forecast = predictor.predict(horizon=horizon, return_df=True)

    if save_path:
        predictor.save(save_path)

    logger.info(f"Previsão gerada: {forecast.shape}")
    return predictor, forecast


# RegimeAwareLSTMPredictor

class RegimeAwareLSTMPredictor:
    """
    LSTM separado por regime de mercado
    Treina um LSTMPredictor independente para cada regime detectado
    """

    def __init__(
        self,
        n_regimes: int = 2,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.4,
        difference_input: bool = True,
        arch: str = "lstm",
    ):
        self.n_regimes = n_regimes
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.difference_input = difference_input
        self.arch = arch
        self.predictors_: Dict[int, LSTMPredictor] = {}
        self._is_fitted = False

    def fit(
        self,
        params_df: pd.DataFrame,
        regime_labels: pd.Series,
        epochs: int = 100,
        verbose: bool = True,
    ) -> "RegimeAwareLSTMPredictor":
        """
        Treina um predictor por regime
        """
        common_idx = params_df.index.intersection(regime_labels.index)
        params_aligned = params_df.loc[common_idx]
        labels_aligned = regime_labels.loc[common_idx]

        for regime in range(self.n_regimes):
            mask = labels_aligned == regime
            subset = params_aligned[mask]

            logger.info(
                f"Regime {regime}: {mask.sum()} observações "
                f"({mask.mean()*100:.1f}% do total)"
            )

            if len(subset) < self.seq_len + 20:
                logger.warning(
                    f"Regime {regime} com poucos dados ({len(subset)}). "
                    f"Usando predictor global como fallback."
                )
                self.predictors_[regime] = None
                continue

            pred = LSTMPredictor(
                seq_len=self.seq_len,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                difference_input=self.difference_input,
                arch=self.arch,
            )
            pred.fit(subset, epochs=epochs, verbose=verbose)
            self.predictors_[regime] = pred

        self._global_predictor = LSTMPredictor(
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            difference_input=self.difference_input,
            arch=self.arch,
        )
        self._global_predictor.fit(params_df, epochs=epochs, verbose=False)

        self._is_fitted = True
        return self

    def predict(
        self,
        current_regime: int,
        horizon: int = 1,
        return_df: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Prevê usando o predictor do regime atual
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        predictor = self.predictors_.get(current_regime)
        if predictor is None:
            logger.warning(f"Sem predictor para regime {current_regime}. Usando global.")
            predictor = self._global_predictor

        return predictor.predict(horizon=horizon, return_df=return_df)


# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    np.random.seed(42)
    T = 500
    n_params = 10

    rho_base = np.array([0.3, 0.4, 0.2, 0.5, 0.35, 0.25, 0.45, 0.3, 0.4, 0.2])
    params_sim = np.zeros((T, n_params))
    params_sim[0] = rho_base + np.random.normal(0, 0.02, n_params)

    for t in range(1, T):
        params_sim[t] = (
            0.95 * params_sim[t - 1]
            + 0.05 * rho_base
            + np.random.normal(0, 0.01, n_params)
        )
        params_sim[t] = np.clip(params_sim[t], -0.99, 0.99)

    col_names = [f"rho_{i+1}{j+1}" for i in range(5) for j in range(i+1, 5)]
    params_df = pd.DataFrame(
        params_sim,
        columns=col_names,
        index=pd.date_range("2020-01-01", periods=T, freq="B"),
    )

    print(f"params_df shape: {params_df.shape}")
    print(params_df.head(3))

    split = int(T * 0.85)
    train_df = params_df.iloc[:split]
    test_df  = params_df.iloc[split:]

    # Feature exógena: volatilidade realizada como proxy de regime
    vol_proxy = pd.DataFrame(
        {"vol_realized": params_df.std(axis=1).rolling(20).mean().fillna(0.02).values},
        index=params_df.index,
    )
    train_aug = pd.concat([train_df, vol_proxy.iloc[:split]], axis=1)
    test_aug  = pd.concat([test_df,  vol_proxy.iloc[split:]], axis=1)

    # Treinando LSTM

    print("\n Treinando LSTM ")
    predictor = LSTMPredictor(
        seq_len=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.4,
        difference_input=True,
        arch="lstm",
    )
    predictor.fit(train_aug, epochs=100, patience=15, verbose=True)

    forecast = predictor.predict(horizon=5)
    forecast = forecast[[c for c in forecast.columns if c != "vol_realized"]]
    print(f"\nPrevisão (próximos 5 dias):\n{forecast.round(4)}")

    metrics = predictor.evaluate(test_aug)
    print(f"\nMétricas no teste: {metrics}")

    history = predictor.get_training_history()
    print(f"\nHistórico (últimas 5 épocas):\n{history.tail()}")

    # Wrapper

    print("\n fit_predict_copula_params (wrapper) ")
    pred2, forecast2 = fit_predict_copula_params(
        params_df=train_df,
        seq_len=20,
        horizon=1,
        arch="gru",
        epochs=50,
        verbose=False,
    )
    print(f"GRU forecast (1 dia): {forecast2.round(4)}")

    # RegimeAware

    print("\n RegimeAwareLSTMPredictor ")
    regime_labels = pd.Series(
        (params_sim[:split, 0] > 0.35).astype(int),
        index=train_df.index,
    )
    print(f"Regime 0: {(regime_labels==0).sum()} obs  Regime 1: {(regime_labels==1).sum()} obs")

    ra = RegimeAwareLSTMPredictor(n_regimes=2, seq_len=20)
    ra.fit(train_df, regime_labels, epochs=50, verbose=False)
    forecast_regime = ra.predict(current_regime=1, horizon=1)
    print(f"Previsão regime 1: {forecast_regime.round(4)}")

    print("\n Todos os testes concluídos.")