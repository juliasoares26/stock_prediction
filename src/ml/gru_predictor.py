import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch não disponível, GRUPredictor usará fallback sklearn")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Importa utilitários compartilhados do lstm_predictor
try:
    from ml.lstm_predictor import build_sequences, LSTMPredictor
except ImportError:
    try:
        from lstm_predictor import build_sequences, LSTMPredictor
    except ImportError:
        # Fallback: redefine build_sequences localmente
        def build_sequences(data, seq_len, horizon=1):
            T, n_features = data.shape
            X_list, y_list = [], []
            for i in range(T - seq_len - horizon + 1):
                X_list.append(data[i: i + seq_len])
                y_window = data[i + seq_len: i + seq_len + horizon]
                y_list.append(y_window[0] if horizon == 1 else y_window)
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


if TORCH_AVAILABLE:

    class _GRUVanilla(nn.Module):

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.2,
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
            self.norm = nn.LayerNorm(hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            out = self.norm(out[:, -1, :])
            out = self.dropout(out)
            return self.fc(out)


    class _GRUWithAttention(nn.Module):
        """
        A atenção permite que o modelo pondere explicitamente quais dias da janela são mais relevantes para a previsão, útil quando choques de volatilidade criam padrões assimétricos
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.2,
            attention_heads: int = 4,
        ):
            super().__init__()
            self.hidden_size = hidden_size

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Scaled dot-product attention
            # Garante que hidden_size seja divisível por attention_heads
            actual_heads = attention_heads
            while hidden_size % actual_heads != 0 and actual_heads > 1:
                actual_heads -= 1

            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=actual_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout)

            # Feed-forward após atenção
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
            )

            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # GRU sobre toda a sequência
            gru_out, _ = self.gru(x)                    

            # Self-attention
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            gru_out = self.norm1(gru_out + attn_out)      # residual

            # Feed-forward
            ff_out  = self.ff(gru_out)
            gru_out = self.norm2(gru_out + ff_out)        # residual

            # Pooling: último timestep + mean pooling
            last    = gru_out[:, -1, :]
            pooled  = gru_out.mean(dim=1)
            out     = self.dropout(last + pooled)

            return self.fc(out)


    class _TCN(nn.Module):
        """
        Temporal Convolutional Network
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout: float = 0.2,
            kernel_size: int = 3,
        ):
            super().__init__()
            layers = []
            in_ch = input_size

            for i in range(num_layers):
                dilation = 2 ** i
                padding  = (kernel_size - 1) * dilation  # causal padding
                out_ch   = hidden_size

                layers.append(_TCNBlock(
                    in_ch, out_ch, kernel_size, dilation, padding, dropout
                ))
                in_ch = out_ch

            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, input_size) → (B, input_size, T)
            x = x.permute(0, 2, 1)
            out = self.network(x)  # (B, hidden_size, T)
            out = out[:, :, -1]    # último timestep
            return self.fc(out)


    class _TCNBlock(nn.Module):
        """Bloco residual causal dilatado para TCN"""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            padding: int,
            dropout: float,
        ):
            super().__init__()
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=padding,
            ))
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(
                out_channels, out_channels, kernel_size,
                dilation=dilation, padding=padding,
            ))
            self.dropout = nn.Dropout(dropout)
            self.relu    = nn.GELU()

            # Projeção residual se canais diferem
            self.residual = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels else None
            )
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self.relu(self.norm1(self.conv1(x)[:, :, :x.size(2)]))
            out = self.dropout(out)
            out = self.relu(self.norm2(self.conv2(out)[:, :, :x.size(2)]))
            out = self.dropout(out)
            res = self.residual(x) if self.residual is not None else x
            return self.relu(out + res)


class GRUPredictor:
    """
    Previsão de parâmetros de cópulas via GRU com variantes de arquitetura

    Interface idêntica ao LSTMPredictor para uso transparente no EnsemblePredictor
    """

    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        arch: str = "attention",
        mc_dropout_samples: int = 50,
        scaler: str = "standard",
        device: str = "auto",
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.arch = arch.lower()
        self.mc_dropout_samples = mc_dropout_samples
        self.scaler_type = scaler
        self.random_state = random_state

        if TORCH_AVAILABLE:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ) if device == "auto" else torch.device(device)
        else:
            self.device = None

        self.model = None
        self.scaler = None
        self.feature_names_: List[str] = []
        self.n_features_: int = 0
        self.train_history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted = False
        self._data_scaled: Optional[np.ndarray] = None
        self._fallback_model = None


    # Preparação de dados


    def _prepare_data(
        self, params: Union[pd.DataFrame, np.ndarray, "pd.Series"]
    ) -> np.ndarray:
        if isinstance(params, pd.Series):
            params = params.to_frame()
        if isinstance(params, pd.DataFrame):
            self.feature_names_ = list(params.columns)
            data = params.values.astype(np.float32)
        else:
            data = np.asarray(params, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self.feature_names_ = [f"param_{i}" for i in range(data.shape[1])]

        # Forward fill NaN
        mask = np.any(np.isnan(data), axis=1)
        if mask.any():
            df_tmp = pd.DataFrame(data).ffill().bfill()
            data = df_tmp.values.astype(np.float32)

        self.n_features_ = data.shape[1]
        return data

    def _init_scaler(self):
        if not SKLEARN_AVAILABLE:
            return None
        return StandardScaler() if self.scaler_type == "standard" else None

    def _build_model(self) -> "nn.Module":
        """Instancia a arquitetura correta"""
        kwargs = dict(
            input_size=self.n_features_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.n_features_,
            dropout=self.dropout,
        )
        if self.arch == "vanilla":
            return _GRUVanilla(**kwargs)
        elif self.arch == "attention":
            return _GRUWithAttention(**kwargs, attention_heads=4)
        elif self.arch == "tcn":
            return _TCN(**kwargs, kernel_size=3)
        else:
            raise ValueError(f"arch desconhecido: {self.arch}. Use 'vanilla', 'attention' ou 'tcn'.")
    
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
    ) -> "GRUPredictor":
        """
        Treina o GRU nos parâmetros históricos de cópula
        """
        np.random.seed(self.random_state)
        if TORCH_AVAILABLE:
            torch.manual_seed(self.random_state)

        data = self._prepare_data(params)
        T = data.shape[0]

        if T < self.seq_len + 10:
            raise ValueError(
                f"Dados insuficientes: T={T}, seq_len={self.seq_len}."
            )

        self.scaler = self._init_scaler()
        data_scaled = (
            self.scaler.fit_transform(data) if self.scaler else data.copy()
        )

        if not TORCH_AVAILABLE:
            return self._fit_fallback(data_scaled, val_split, verbose)

        X, y = build_sequences(data_scaled, self.seq_len, horizon=1)
        n_val   = max(1, int(len(X) * val_split))
        n_train = len(X) - n_val

        X_train = torch.tensor(X[:n_train], dtype=torch.float32)
        y_train = torch.tensor(y[:n_train], dtype=torch.float32)
        X_val   = torch.tensor(X[n_train:], dtype=torch.float32)
        y_val   = torch.tensor(y[n_train:], dtype=torch.float32)

        train_dl = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        self.model = self._build_model().to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        criterion = nn.HuberLoss(delta=1.0)  

        best_val  = np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            #  Treino 
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

            #  Validação 
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val.to(self.device))
                val_loss = criterion(val_pred, y_val.to(self.device)).item()

            train_loss = np.mean(train_losses)
            self.train_history_["train_loss"].append(train_loss)
            self.train_history_["val_loss"].append(val_loss)
            scheduler.step()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = deepcopy({k: v.cpu() for k, v in self.model.state_dict().items()})
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                logger.info(
                    f"[GRU-{self.arch}] Epoch {epoch:4d}/{epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}  "
                    f"patience={patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                logger.info(f"Early stopping na época {epoch}.")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        self._data_scaled = data_scaled
        self._is_fitted = True
        logger.info(
            f"GRUPredictor treinado arch={self.arch}  "
            f"d={self.n_features_} device={self.device}  "
            f"best_val={best_val:.6f}"
        )
        return self

    def _fit_fallback(self, data_scaled, val_split, verbose):
        """Fallback ElasticNet para ambientes sem PyTorch"""
        X, y = build_sequences(data_scaled, self.seq_len, horizon=1)
        n_train = int(len(X) * (1 - val_split))
        X_flat  = X.reshape(len(X), -1)
        self._fallback_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        self._fallback_model.fit(X_flat[:n_train], y[:n_train])
        if verbose:
            val_pred = self._fallback_model.predict(X_flat[n_train:])
            mse = mean_squared_error(y[n_train:], val_pred)
            logger.info(f"[GRU fallback ElasticNet] val MSE={mse:.6f}")
        self._data_scaled = data_scaled
        self._is_fitted   = True
        return self
    
    # Predict

    def predict(
        self,
        horizon: int = 1,
        return_df: bool = True,
        last_window: Optional[np.ndarray] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Prevê os próximos 'horizon' passos

        Interface idêntica ao LSTMPredictor.predict() para uso no Ensemble
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        window = (
            self._data_scaled[-self.seq_len:].copy()
            if last_window is None
            else (self.scaler.transform(last_window) if self.scaler else last_window.copy())
        )

        predictions_scaled = []

        for _ in range(horizon):
            inp = window[-self.seq_len:]

            if TORCH_AVAILABLE and self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    x_t = torch.tensor(inp[np.newaxis], dtype=torch.float32).to(self.device)
                    pred = self.model(x_t).cpu().numpy()[0]
            elif self._fallback_model:
                pred = self._fallback_model.predict(inp.reshape(1, -1))[0]
            else:
                raise RuntimeError("Nenhum modelo disponível.")

            predictions_scaled.append(pred)
            window = np.vstack([window, pred])

        preds = np.array(predictions_scaled, dtype=np.float32)
        if self.scaler:
            preds = self.scaler.inverse_transform(preds)

        if return_df:
            return pd.DataFrame(preds, columns=self.feature_names_)
        return preds

    
    # Incerteza via MC Dropout
    
    def predict_with_uncertainty(
        self,
        horizon: int = 1,
        n_samples: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Estimativa de incerteza via Monte Carlo Dropout

        Mantém dropout ativo durante inferência e faz 'n_samples' previsões para obter distribuição empírica do forecast

        """
        if not self._is_fitted or not TORCH_AVAILABLE or self.model is None:
            logger.warning("MC Dropout requer PyTorch, Retornando previsão pontual")
            pred = self.predict(horizon=horizon, return_df=True)
            zeros = pd.DataFrame(
                np.zeros_like(pred.values), columns=self.feature_names_
            )
            return {"mean": pred, "std": zeros, "lower": pred, "upper": pred, "samples": None}

        n_mc = n_samples or self.mc_dropout_samples

        # Ativar dropout durante inferência
        def _enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.eval()
        self.model.apply(_enable_dropout)

        all_samples = []
        window = self._data_scaled[-self.seq_len:].copy()

        with torch.no_grad():
            for _ in range(n_mc):
                w = window.copy()
                sample_preds = []
                for _ in range(horizon):
                    inp = w[-self.seq_len:]
                    x_t = torch.tensor(inp[np.newaxis], dtype=torch.float32).to(self.device)
                    p   = self.model(x_t).cpu().numpy()[0]
                    sample_preds.append(p)
                    w = np.vstack([w, p])
                all_samples.append(sample_preds)

        self.model.eval()  # desativar dropout

        samples = np.array(all_samples, dtype=np.float32)  # (n_mc, horizon, d)

        if self.scaler:
            orig_shape = samples.shape
            samples_flat = samples.reshape(-1, self.n_features_)
            samples_inv  = self.scaler.inverse_transform(samples_flat)
            samples = samples_inv.reshape(orig_shape)

        mean_pred = samples.mean(axis=0)
        std_pred = samples.std(axis=0)
        lower = np.percentile(samples, 5,  axis=0)
        upper = np.percentile(samples, 95, axis=0)

        def _to_df(arr):
            return pd.DataFrame(arr, columns=self.feature_names_)

        return {
            "mean": _to_df(mean_pred),
            "std":   _to_df(std_pred),
            "lower": _to_df(lower),
            "upper":  _to_df(upper),
            "samples": samples,
        }

    # Avaliação

    def evaluate(
        self,
        params_test: Union[pd.DataFrame, np.ndarray],
        metrics: List[str] = ["mse", "mae", "rmse"],
    ) -> Dict[str, float]:
        """Avalia em dados de teste"""
        data = self._prepare_data(params_test)
        data_scaled = self.scaler.transform(data) if self.scaler else data.copy()
        X, y_true = build_sequences(data_scaled, self.seq_len, horizon=1)

        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                y_pred_s = self.model(
                    torch.tensor(X, dtype=torch.float32).to(self.device)
                ).cpu().numpy()
        elif self._fallback_model:
            y_pred_s = self._fallback_model.predict(X.reshape(len(X), -1))
        else:
            raise RuntimeError("Modelo não treinado.")

        if self.scaler:
            y_pred = self.scaler.inverse_transform(y_pred_s)
            y_true_ = self.scaler.inverse_transform(y_true)
        else:
            y_pred, y_true_ = y_pred_s, y_true

        out = {}
        if "mse"  in metrics: out["mse"] = float(np.mean((y_true_ - y_pred)**2))
        if "mae"  in metrics: out["mae"] = float(np.mean(np.abs(y_true_ - y_pred)))
        if "rmse" in metrics: out["rmse"] = float(np.sqrt(np.mean((y_true_ - y_pred)**2)))
        if "mape" in metrics:
            mask = y_true_ != 0
            out["mape"] = (
                float(np.mean(np.abs((y_true_[mask] - y_pred[mask]) / y_true_[mask])) * 100)
                if mask.any() else np.nan
            )

        logger.info(f"[GRU-{self.arch}] Avaliação: {out}")
        return out

    # Save

    def save(self, path: Union[str, Path]):
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "config": {
                "seq_len": self.seq_len, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "arch": self.arch, "mc_dropout_samples": self.mc_dropout_samples,
                "scaler": self.scaler_type, "random_state": self.random_state,
            },
            "feature_names": self.feature_names_,
            "n_features": self.n_features_,
            "scaler": self.scaler,
            "train_history": self.train_history_,
            "data_scaled": self._data_scaled,
            "is_fitted": self._is_fitted,
        }
        if TORCH_AVAILABLE and self.model:
            state["model_state_dict"] = self.model.state_dict()
            state["backend"] = "torch"
        else:
            state["fallback_model"] = self._fallback_model
            state["backend"] = "sklearn"

        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"GRUPredictor salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GRUPredictor":
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(**state["config"])
        obj.feature_names_ = state["feature_names"]
        obj.n_features_  = state["n_features"]
        obj.scaler = state["scaler"]
        obj.train_history_ = state["train_history"]
        obj._is_fitted = state["is_fitted"]
        if state.get("data_scaled") is not None:
            obj._data_scaled = state["data_scaled"]
        if state["backend"] == "torch" and TORCH_AVAILABLE:
            obj.model = obj._build_model().to(obj.device)
            obj.model.load_state_dict(state["model_state_dict"])
        else:
            obj._fallback_model = state.get("fallback_model")
        logger.info(f"GRUPredictor carregado de {path}")
        return obj

    def get_training_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.train_history_)

# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    
    np.random.seed(42)
    T, d = 500, 10
    rho_base = np.array([0.3, 0.4, 0.2, 0.5, 0.35, 0.25, 0.45, 0.3, 0.4, 0.2])
    params_sim = np.zeros((T, d))
    params_sim[0] = rho_base
    for t in range(1, T):
        params_sim[t] = np.clip(
            0.95 * params_sim[t-1] + 0.05 * rho_base + np.random.normal(0, 0.01, d),
            -0.99, 0.99
        )

    cols = [f"rho_{i+1}{j+1}" for i in range(5) for j in range(i+1, 5)]
    params_df = pd.DataFrame(params_sim, columns=cols,
                             index=pd.date_range("2020-01-01", periods=T, freq="B"))

    split = int(T * 0.85)
    train_df, test_df = params_df.iloc[:split], params_df.iloc[split:]

    for arch in ["vanilla", "attention", "tcn"]:
        print(f"\n GRU arch={arch} ")
        gru = GRUPredictor(seq_len=40, hidden_size=32, num_layers=2, arch=arch)
        gru.fit(train_df, epochs=30, patience=8, verbose=False)

        forecast = gru.predict(horizon=5, return_df=True)
        print(f"Forecast (5 dias):\n{forecast.round(4)}")

        metrics = gru.evaluate(test_df)
        print(f"Métricas: {metrics}")

    # MC Dropout
    print("\n MC Dropout (incerteza) ")
    gru_att = GRUPredictor(seq_len=40, hidden_size=32, arch="attention", mc_dropout_samples=30)
    gru_att.fit(train_df, epochs=30, patience=8, verbose=False)
    unc = gru_att.predict_with_uncertainty(horizon=1)
    print(f"Média: {unc['mean'].round(4).to_string()}")
    print(f"Desvio: {unc['std'].round(4).to_string()}")
    print(f"IC 90%: [{unc['lower'].values[0,0]:.4f}, {unc['upper'].values[0,0]:.4f}]")

    print("\n Todos os testes concluídos.")
