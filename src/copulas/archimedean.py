import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, Union
from scipy import stats
from scipy.optimize import minimize_scalar, brentq


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Classe base 
class _ArchimedeanBase:
    """Interface comum para todas as cópulas arquimedianas."""

    def __init__(self):
        self.theta_: float = np.nan
        self.log_likelihood_: float = np.nan
        self.aic_: float = np.nan
        self.bic_: float = np.nan
        self._is_fitted = False

    def fit(self, u: np.ndarray, v: np.ndarray) -> "_ArchimedeanBase":
        raise NotImplementedError

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """CDF condicional h(u|v) = ∂C(u,v)/∂v — usada em vine_copulas.py."""
        raise NotImplementedError

    def h_inverse(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inversa da h-function"""
        result = np.zeros_like(u)
        for i, (ui, vi) in enumerate(zip(u.ravel(), v.ravel())):
            try:
                result[i] = brentq(
                    lambda x: self.h_function(np.array([x]), np.array([vi]))[0] - ui,
                    1e-9, 1-1e-9, xtol=1e-9, maxiter=100
                )
            except Exception:
                result[i] = 0.5
        return result

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulação via inversão da h-function """
        if seed is not None:
            np.random.seed(seed)
        u = np.random.uniform(0, 1, n_sim)
        w = np.random.uniform(0, 1, n_sim)
        v = self.h_inverse(w, u)
        return np.column_stack([u, v]).astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        raise NotImplementedError

    def _compute_aic_bic(self, ll: float, T: int, k: int = 1):
        self.log_likelihood_ = ll
        self.aic_ = -2*ll + 2*k
        self.bic_ = -2*ll + k*np.log(T)

    @staticmethod
    def _clean(u: np.ndarray) -> np.ndarray:
        return np.clip(u.ravel().astype(np.float64), 1e-9, 1-1e-9)

    @staticmethod
    def _kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
        tau, _ = stats.kendalltau(u, v)
        return float(tau)


# Clayton

class ClaytonCopula(_ArchimedeanBase):
    """
    Cópula Clayton: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

    θ > 0 (θ → 0: independência, θ → ∞: comonotonicidade)
    τ = θ/(θ+2)  →  θ = 2τ/(1-τ)

    Cauda inferior: λ_L = 2^{-1/θ}
    Cauda superior: λ_U = 0
    """

    def __init__(self, theta_bounds: Tuple[float, float] = (0.01, 50.0)):
        super().__init__()
        self.theta_bounds = theta_bounds

    def fit(self, u: np.ndarray, v: np.ndarray) -> "ClaytonCopula":
        u, v = self._clean(u), self._clean(v)
        T = len(u)

        # Estimativa inicial via Kendall
        tau = self._kendall_tau(u, v)
        theta0 = max(2*tau/(1-tau), 0.1) if tau > 0 else 0.5

        def neg_ll(theta):
            if theta <= 0:
                return 1e10
            ll = self._log_density_vals(u, v, theta)
            return -np.sum(ll[np.isfinite(ll)])

        res = minimize_scalar(neg_ll, bounds=self.theta_bounds, method="bounded")
        self.theta_ = float(res.x)
        self._compute_aic_bic(-res.fun, T)
        self._is_fitted = True

        logger.debug(f"Clayton | θ={self.theta_:.4f} | τ_implied={self.theta_/(self.theta_+2):.3f}")
        return self

    def _log_density_vals(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        """Log-densidade c(u,v;θ) para Clayton."""
        log_c = (
            np.log(1 + theta)
            + (-1 - theta) * (np.log(u) + np.log(v))
            + (-1/theta - 2) * np.log(u**(-theta) + v**(-theta) - 1)
        )
        return log_c

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        return np.maximum(u**(-self.theta_) + v**(-self.theta_) - 1, 1e-10)**(-1/self.theta_)

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        return self._log_density_vals(u, v, self.theta_)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        h(u|v) = ∂C/∂v = v^{-(θ+1)} · (u^{-θ} + v^{-θ} - 1)^{-(1/θ+1)}
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        inner = np.maximum(u**(-theta) + v**(-theta) - 1, 1e-10)
        return v**(-(theta+1)) * inner**(-(1/theta + 1))

    def h_inverse(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Inversão analítica da h-function de Clayton (mais eficiente que brentq).
        x = ((u · v^{θ+1})^{-θ/(θ+1)} + 1 - v^{-θ})^{-1/θ}
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        a = (u * v**(theta+1))**(-theta/(theta+1)) + 1 - v**(-theta)
        return np.maximum(a, 1e-10)**(-1/theta)

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulacao Clayton via inversao condicional (h_inverse analitica)
        Garante marginais U(0,1) 
        """
        if seed is not None:
            np.random.seed(seed)
        theta = self.theta_
        u = np.random.uniform(0, 1, n_sim)
        w = np.random.uniform(0, 1, n_sim)
        inner = (w * u**(theta+1))**(-theta/(theta+1)) + 1 - u**(-theta)
        v = np.maximum(inner, 1e-10)**(-1/theta)
        return np.column_stack([
            np.clip(u, 1e-9, 1-1e-9),
            np.clip(v, 1e-9, 1-1e-9)
        ]).astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        theta = self.theta_
        return {
            "lower_tail": float(2**(-1/theta)),
            "upper_tail": 0.0,
        }


# Gumbel

class GumbelCopula(_ArchimedeanBase):
    """
    Cópula Gumbel: C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ})

    θ ≥ 1 (θ=1: independência, θ→∞: comonotonicidade)
    τ = 1 - 1/θ  →  θ = 1/(1-τ)

    Cauda inferior: λ_L = 0
    Cauda superior: λ_U = 2 - 2^{1/θ}
    """

    def __init__(self, theta_bounds: Tuple[float, float] = (1.001, 50.0)):
        super().__init__()
        self.theta_bounds = theta_bounds

    def fit(self, u: np.ndarray, v: np.ndarray) -> "GumbelCopula":
        u, v = self._clean(u), self._clean(v)
        T = len(u)

        tau = self._kendall_tau(u, v)
        theta0 = max(1/(1 - tau), 1.1) if tau > 0 else 1.5

        def neg_ll(theta):
            if theta < 1:
                return 1e10
            ll = self._log_density_vals(u, v, theta)
            return -np.sum(ll[np.isfinite(ll)])

        res = minimize_scalar(neg_ll, bounds=self.theta_bounds, method="bounded")
        self.theta_ = float(res.x)
        self._compute_aic_bic(-res.fun, T)
        self._is_fitted = True

        logger.debug(f"Gumbel | θ={self.theta_:.4f} | τ_implied={1-1/self.theta_:.3f}")
        return self

    def _log_density_vals(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        A = (-np.log(u))**theta + (-np.log(v))**theta
        A = np.maximum(A, 1e-300)
        C = np.exp(-A**(1/theta))
        log_c = (
            np.log(C)
            + (1/theta - 2) * np.log(A)
            + (theta-1) * (np.log(-np.log(u)) + np.log(-np.log(v)))
            - np.log(u) - np.log(v)
            + np.log(A**(1/theta) + theta - 1)
        )
        return log_c

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        A = (-np.log(u))**theta + (-np.log(v))**theta
        return np.exp(-A**(1/theta))

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        return self._log_density_vals(u, v, self.theta_)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        h(u|v) = C(u,v) · A^{1/θ - 1} · (-ln v)^{θ-1} / v
        onde A = (-ln u)^θ + (-ln v)^θ
        """
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        A = np.maximum((-np.log(u))**theta + (-np.log(v))**theta, 1e-300)
        C = np.exp(-A**(1/theta))
        h = C * A**(1/theta - 1) * (-np.log(v))**(theta-1) / v
        return np.clip(h, 1e-9, 1-1e-9)

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulação de Gumbel via Stable distribution frailty."""
        if seed is not None:
            np.random.seed(seed)
        # Método de Marshall-Olkin com stable frailty
        theta = self.theta_
        # Stable(1/theta) rv via Chambers-Mallows-Stuck
        alpha = 1/theta
        u_s = np.random.uniform(0, np.pi, n_sim)
        e_s = np.random.exponential(1, n_sim)
        stable = (
            np.sin(alpha * u_s) / (np.sin(u_s))**(1/alpha)
            * (np.sin((1-alpha)*u_s) / e_s)**((1-alpha)/alpha)
        )
        stable = np.maximum(stable, 1e-10)
        e1 = np.random.exponential(1, n_sim)
        e2 = np.random.exponential(1, n_sim)
        u = np.exp(-(e1/stable)**(1/theta))
        v = np.exp(-(e2/stable)**(1/theta))
        return np.column_stack([
            np.clip(u, 1e-9, 1-1e-9),
            np.clip(v, 1e-9, 1-1e-9)
        ]).astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        theta = self.theta_
        return {
            "lower_tail": 0.0,
            "upper_tail": float(2 - 2**(1/theta)),
        }

# Frank

class FrankCopula(_ArchimedeanBase):
    """
    Cópula Frank: C(u,v;θ) = -1/θ · ln[1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1)]
    """

    def __init__(self, theta_bounds: Tuple[float, float] = (-50.0, 50.0)):
        super().__init__()
        self.theta_bounds = theta_bounds

    def fit(self, u: np.ndarray, v: np.ndarray) -> "FrankCopula":
        u, v = self._clean(u), self._clean(v)
        T = len(u)

        tau = self._kendall_tau(u, v)
        # Estimativa inicial via relação tau-theta de Frank 
        theta0 = 5.0 * np.sign(tau) if abs(tau) > 0.01 else 0.5

        def neg_ll(theta):
            if abs(theta) < 1e-6:
                return 1e10
            ll = self._log_density_vals(u, v, theta)
            return -np.sum(ll[np.isfinite(ll)])

        res = minimize_scalar(neg_ll, bounds=self.theta_bounds, method="bounded")
        self.theta_ = float(res.x)
        self._compute_aic_bic(-res.fun, T)
        self._is_fitted = True

        logger.debug(f"Frank | θ={self.theta_:.4f}")
        return self

    def _log_density_vals(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        if abs(theta) < 1e-6:
            return np.zeros(len(u))

        et = np.exp(-theta)
        eu = np.exp(-theta*u)
        ev = np.exp(-theta*v)

        numer = -theta * et * (eu - 1) * (ev - 1)
        denom = (et - 1 + (eu - 1)*(ev - 1))
        denom = np.maximum(np.abs(denom), 1e-300) * np.sign(denom + 1e-300)

        log_c = np.log(np.maximum(np.abs(numer), 1e-300)) - 2*np.log(np.maximum(np.abs(denom), 1e-300))
        return log_c

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        if abs(theta) < 1e-6:
            return u * v
        et = np.exp(-theta)
        num = (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1)
        den = et - 1
        return -1/theta * np.log(1 + num/np.maximum(den, 1e-300))

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        return self._log_density_vals(u, v, self.theta_)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = (e^{-θv}-1)(e^{-θu}-1) / [(e^{-θ}-1) + (e^{-θv}-1)(e^{-θu}-1)]"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        if abs(theta) < 1e-6:
            return u.copy()
        eu = np.exp(-theta*u)
        ev = np.exp(-theta*v)
        et = np.exp(-theta)
        num = (eu - 1) * ev
        den = (et - 1) + (eu - 1)*(ev - 1)
        h = num / np.where(np.abs(den) < 1e-300, 1e-300, den)
        return np.clip(h, 1e-9, 1-1e-9)

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulação de Frank via inversão condicional analítica"""
        if seed is not None:
            np.random.seed(seed)
        theta = self.theta_
        u = np.random.uniform(0, 1, n_sim)
        w = np.random.uniform(0, 1, n_sim)

        if abs(theta) < 1e-6:
            return np.column_stack([u, w]).astype(np.float32)

        et = np.exp(-theta)
        eu = np.exp(-theta*u)
        # Inversão analítica da h-function
        v = -1/theta * np.log(1 + w*(et-1) / (w*(eu-1) - eu + 1 + 1e-300))
        return np.column_stack([
            np.clip(u, 1e-9, 1-1e-9),
            np.clip(v, 1e-9, 1-1e-9)
        ]).astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        return {"lower_tail": 0.0, "upper_tail": 0.0}

# Joe

class JoeCopula(_ArchimedeanBase):
    """
    Cópula Joe: C(u,v;θ) = 1 - [(1-u)^θ + (1-v)^θ - (1-u)^θ(1-v)^θ]^{1/θ}
    """

    def __init__(self, theta_bounds: Tuple[float, float] = (1.001, 50.0)):
        super().__init__()
        self.theta_bounds = theta_bounds

    def fit(self, u: np.ndarray, v: np.ndarray) -> "JoeCopula":
        u, v = self._clean(u), self._clean(v)
        T = len(u)

        tau = self._kendall_tau(u, v)
        theta0 = max(1.5, 1 + tau)

        def neg_ll(theta):
            if theta < 1:
                return 1e10
            ll = self._log_density_vals(u, v, theta)
            return -np.sum(ll[np.isfinite(ll)])

        res = minimize_scalar(neg_ll, bounds=self.theta_bounds, method="bounded")
        self.theta_ = float(res.x)
        self._compute_aic_bic(-res.fun, T)
        self._is_fitted = True

        logger.debug(f"Joe | θ={self.theta_:.4f}")
        return self

    def _log_density_vals(self, u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
        a = (1-u)**theta
        b = (1-v)**theta
        A = np.maximum(a + b - a*b, 1e-300)
        log_c = (
            np.log(theta-1+A**(1/theta))
            + (1/theta - 2)*np.log(A)
            + (theta-1)*(np.log(1-u) + np.log(1-v))
        )
        return log_c

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        a = (1-u)**theta
        b = (1-v)**theta
        return 1 - np.maximum(a + b - a*b, 1e-300)**(1/theta)

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        return self._log_density_vals(u, v, self.theta_)

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """h(u|v) = ∂C/∂v = (1-v)^{θ-1}(1-(1-u)^θ) · A^{1/θ-1}"""
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        u, v = self._clean(u), self._clean(v)
        theta = self.theta_
        a = (1-u)**theta
        b = (1-v)**theta
        A = np.maximum(a + b - a*b, 1e-300)
        h = (1-v)**(theta-1) * (1-a) * A**(1/theta-1)
        return np.clip(h, 1e-9, 1-1e-9)

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulação via inversão numérica"""
        if seed is not None:
            np.random.seed(seed)
        u = np.random.uniform(0, 1, n_sim)
        w = np.random.uniform(0, 1, n_sim)
        v = np.array([
            brentq(
                lambda x: self.h_function(np.array([u[i]]), np.array([x]))[0] - w[i],
                1e-9, 1-1e-9, xtol=1e-8
            )
            for i in range(n_sim)
        ])
        return np.column_stack([u, v]).astype(np.float32)

    def tail_dependence(self) -> Dict[str, float]:
        theta = self.theta_
        return {
            "lower_tail": 0.0,
            "upper_tail": float(2 - 2**(1/theta)),
        }

# Cópula Clayton Rotacionada 180°

class ClaytonCopula180(_ArchimedeanBase):
    """
    Clayton rotacionada 180°, captura dependência de CAUDA SUPERIOR.
    C(u,v) = u + v - 1 + Clayton(1-u, 1-v)
    λ_L = 0,  λ_U = 2^{-1/θ}
    """

    def __init__(self):
        super().__init__()
        self._base = ClaytonCopula()

    def fit(self, u: np.ndarray, v: np.ndarray) -> "ClaytonCopula180":
        u, v = self._clean(u), self._clean(v)
        self._base.fit(1-u, 1-v)
        self.theta_ = self._base.theta_
        self.log_likelihood_ = self._base.log_likelihood_
        self.aic_ = self._base.aic_
        self.bic_ = self._base.bic_
        self._is_fitted = True
        return self

    def h_function(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        h_base = self._base.h_function(1-u, 1-v)
        return 1 - h_base

    def log_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u, v = self._clean(u), self._clean(v)
        return self._base.log_density(1-u, 1-v)

    def simulate(self, n_sim: int, seed: Optional[int] = None) -> np.ndarray:
        sim = self._base.simulate(n_sim, seed)
        return 1 - sim

    def tail_dependence(self) -> Dict[str, float]:
        theta = self.theta_
        return {
            "lower_tail": 0.0,
            "upper_tail": float(2**(-1/theta)),
        }


COPULA_FAMILIES = {
    "clayton":     ClaytonCopula,
    "gumbel":      GumbelCopula,
    "frank":       FrankCopula,
    "joe":         JoeCopula,
    "clayton180":  ClaytonCopula180,
}


def get_copula(family: str) -> _ArchimedeanBase:
    family = family.lower()
    if family not in COPULA_FAMILIES:
        raise ValueError(f"Família desconhecida: {family}. Opções: {list(COPULA_FAMILIES.keys())}")
    return COPULA_FAMILIES[family]()

# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("Teste: Cópulas Arquimedianas")

    np.random.seed(42)
    n = 1000

    # Simular Clayton com theta=2 via inversao condicional 
    true_theta = 2.0
    u_raw = np.random.uniform(0, 1, n)
    w_raw = np.random.uniform(0, 1, n)

    # h_inverse analitica de Clayton
    _inner = (w_raw * u_raw**(true_theta+1))**(-true_theta/(true_theta+1)) + 1 - u_raw**(-true_theta)
    v_raw  = np.maximum(_inner, 1e-10)**(-1/true_theta)
    u_clay = np.clip(u_raw, 1e-9, 1-1e-9)
    v_clay = np.clip(v_raw, 1e-9, 1-1e-9)

    # Verificar marginais e tau
    from scipy.stats import kstest as _ks, kendalltau as _kt
    _ku, _pu = _ks(u_clay, 'uniform')
    _tau, _  = _kt(u_clay, v_clay)
    print(f'Dados Clayton: KS marginal U p={_pu:.3f} tau={_tau:.3f} theta_tau={2*_tau/(1-_tau):.3f}')

    print(f"\n Clayton (θ verdadeiro={true_theta})")
    clay = ClaytonCopula()
    clay.fit(u_clay, v_clay)
    print(f"θ estimado: {clay.theta_:.4f} AIC={clay.aic_:.2f}")
    print(f"Tail dep: {clay.tail_dependence()}")

    sim_clay = clay.simulate(500)
    print(f"Simulação: {sim_clay.shape} range [{sim_clay.min():.3f}, {sim_clay.max():.3f}]")

    # Gumbel
    print("\n GumbelCopula")
    gum = GumbelCopula()
    gum.fit(u_clay, v_clay)
    print(f"θ={gum.theta_:.4f} AIC={gum.aic_:.2f} Tail: {gum.tail_dependence()}")

    # Frank
    print("\n FrankCopula")
    frank = FrankCopula()
    frank.fit(u_clay, v_clay)
    print(f"θ={frank.theta_:.4f} AIC={frank.aic_:.2f} Tail: {frank.tail_dependence()}")

    # Joe
    print("\n JoeCopula")
    joe = JoeCopula()
    joe.fit(u_clay, v_clay)
    print(f"θ={joe.theta_:.4f} AIC={joe.aic_:.2f}  Tail: {joe.tail_dependence()}")

    # h-function 
    print("\n h-function Clayton ")
    h_vals = clay.h_function(u_clay[:5], v_clay[:5])
    h_inv  = clay.h_inverse(h_vals, v_clay[:5])
    print(f"h(u|v)  = {h_vals.round(4)}")
    print(f"h⁻¹(h)  = {h_inv.round(4)}")
    print(f"u orig  = {u_clay[:5].round(4)}")
    print(f"Erro h⁻¹: {np.abs(h_inv - u_clay[:5]).max():.6f}")

    # Factory
    print("\n Factory ")
    for fam in COPULA_FAMILIES:
        cop = get_copula(fam)
        cop.fit(u_clay, v_clay)
        print(f"{fam:12s}  θ={cop.theta_:.3f}  AIC={cop.aic_:.2f}  {cop.tail_dependence()}")

    print("\n Teste concluído.")
