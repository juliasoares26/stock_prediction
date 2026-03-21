import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from scipy import stats
from scipy.optimize import minimize
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_dependence(original, simulated):
    from scipy.stats import kendalltau
    n_vars = original.shape[1]
    print("comparação kendall's tau")
    print("Par\t\tOriginal\tSimulado")
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            tau_orig, _ = kendalltau(original[:, i], original[:, j])
            tau_sim, _ = kendalltau(simulated[:, i], simulated[:, j])
            print(f"({i},{j})\t\t{tau_orig:.3f}\t\t{tau_sim:.3f}")


def plot_vine_structure(cvine):
    print("\nEstrutura da C-vine:")
    for tree in range(cvine.n_dim - 1):
        print(f"\nÁrvore {tree + 1}:")
        tree_copulas = [(k, v) for k, v in cvine.copulas.items() if k[0] == tree]
        for key, copula in tree_copulas:
            _, v1, v2 = key
            tail_L, tail_U = copula.tail_dependence()
            print(f"Edge ({v1},{v2}): {copula.family}")
            print(f"θ={copula.param:.3f}, λL={tail_L:.3f}, λU={tail_U:.3f}")

# PairCopula

class PairCopula:
    def __init__(self, family: str = 'gaussian'):
        self.family = family.lower()
        self.param = None
        self.param2 = None  # graus de liberdade para t-copula

    def fit(self, u: np.ndarray, v: np.ndarray) -> Dict:
        if self.family == 'gaussian':
            self._fit_gaussian(u, v)
        elif self.family == 't':
            self._fit_t(u, v)
        elif self.family == 'clayton':
            self._fit_clayton(u, v)
        elif self.family == 'gumbel':
            self._fit_gumbel(u, v)
        elif self.family == 'frank':
            self._fit_frank(u, v)
        else:
            raise ValueError(f"familia desconhecida: {self.family}")

        return {
            'family': self.family,
            'param': self.param,
            'param2': self.param2,
            'tau': self.kendall_tau(u, v)
        }

    def _fit_gaussian(self, u, v):
        z_u = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        z_v = stats.norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        self.param = np.corrcoef(z_u, z_v)[0, 1]

    def _fit_t(self, u, v):
        def neg_loglik(params):
            rho, nu = params
            if abs(rho) >= 1 or nu <= 2:
                return 1e10
            z_u = stats.t.ppf(np.clip(u, 1e-10, 1 - 1e-10), nu)
            z_v = stats.t.ppf(np.clip(v, 1e-10, 1 - 1e-10), nu)
            R = np.array([[1, rho], [rho, 1]])
            R_inv = np.linalg.inv(R)
            det_R = np.linalg.det(R)
            ll = 0
            for i in range(len(u)):
                z = np.array([z_u[i], z_v[i]])
                if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                    continue
                from scipy.special import gamma
                ll += (
                    np.log(gamma((nu + 2) / 2) * gamma(nu / 2) / (gamma((nu + 1) / 2) ** 2))
                    - 0.5 * np.log(det_R)
                    + (nu + 1) * np.log(1 + z_u[i] ** 2 / nu)
                    + (nu + 1) * np.log(1 + z_v[i] ** 2 / nu)
                    - (nu + 2) + np.log(1 + z @ R_inv @ z / nu)
                )
            return -ll

        result = minimize(neg_loglik, x0=[0.5, 10],
                          bounds=[(-0.99, 0.99), (2.1, 30)], method='L-BFGS-B')
        self.param = result.x[0]
        self.param2 = result.x[1]

    def _fit_clayton(self, u, v):
        tau = self.kendall_tau(u, v)
        self.param = max(2 * tau / (1 - tau), 0.01)

    def _fit_gumbel(self, u, v):
        tau = self.kendall_tau(u, v)
        self.param = max(1 / (1 - tau), 1.01)

    def _fit_frank(self, u, v):
        tau = self.kendall_tau(u, v)
        if abs(tau) < 0.01:
            self.param = 0.0
            return
        from scipy.integrate import quad

        def debye(theta):
            if abs(theta) < 1e-10:
                return 1.0
            try:
                return quad(lambda t: t / (np.exp(t) - 1), 0, abs(theta))[0] / theta
            except:
                return 1.0

        def objective(theta):
            if abs(theta) < 1e-10:
                return tau ** 2
            try:
                d = debye(theta)
                return (tau - (1 - 4 / theta * (1 - d))) ** 2
            except:
                return 1e10

        result = minimize(objective, x0=5.0, method='Nelder-Mead')
        self.param = result.x[0]

    @staticmethod
    def kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(u, v)
        return tau

    # h-functions vetorizadas aceitam scalar ou ndarray

    def h_function(self, u, v, deriv: int = 1):
        """
        deriv=1 → h(u|v) = ∂C/∂v
        deriv=2 → h(v|u) = ∂C/∂u
        """
        eps = 1e-10
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        scalar_input = np.ndim(u) == 0 and np.ndim(v) == 0

        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))

        if self.family == 'gaussian':
            z_u = stats.norm.ppf(u)
            z_v = stats.norm.ppf(v)
            rho = self.param
            denom = np.sqrt(max(1 - rho ** 2, eps))
            if deriv == 1:
                result = stats.norm.cdf((z_u - rho * z_v) / denom)
            else:
                result = stats.norm.cdf((z_v - rho * z_u) / denom)

        elif self.family == 'clayton':
            theta = self.param
            if deriv == 1:
                term = np.maximum(u ** (-theta) + v ** (-theta) - 1, eps)
                result = np.clip(v ** (-theta - 1) * term ** (-1 / theta - 1), eps, 1 - eps)
            else:
                term = np.maximum(u ** (-theta) + v ** (-theta) - 1, eps)
                result = np.clip(u ** (-theta - 1) * term ** (-1 / theta - 1), eps, 1 - eps)

        elif self.family == 'gumbel':
            theta = self.param
            log_u = np.log(np.maximum(u, eps))
            log_v = np.log(np.maximum(v, eps))
            a = (-log_u) ** theta
            b = (-log_v) ** theta
            sum_ab = np.maximum(a + b, eps)
            sum_pow = sum_ab ** (1 / theta)
            C = np.exp(-sum_pow)

            if deriv == 1:
                b_safe = np.maximum(b, eps)
                term1 = sum_pow / sum_ab
                term2 = b_safe ** (1 - 1 / theta)
                result = np.clip(C * term1 * term2 / np.maximum(v, eps), eps, 1 - eps)
            else:
                a_safe = np.maximum(a, eps)
                term1 = sum_pow / sum_ab
                term2 = a_safe ** (1 - 1 / theta)
                result = np.clip(C * term1 * term2 / np.maximum(u, eps), eps, 1 - eps)

        elif self.family == 'frank':
            theta = self.param
            if abs(theta) < eps:
                result = v.copy() if deriv == 1 else u.copy()
            else:
                tu = np.clip(-theta * u, -700, 700)
                tv = np.clip(-theta * v, -700, 700)
                tn = np.clip(-theta, -700, 700)
                exp_u = np.exp(tu)
                exp_v = np.exp(tv)
                exp_n = np.exp(tn)
                denom = np.where(
                    np.abs((exp_n - 1) + (exp_u - 1) * (exp_v - 1)) < eps,
                    eps,
                    (exp_n - 1) + (exp_u - 1) * (exp_v - 1)
                )
                if deriv == 1:
                    result = np.clip((exp_v - 1) / denom, eps, 1 - eps)
                else:
                    result = np.clip((exp_u - 1) / denom, eps, 1 - eps)

        else:
            # Fallback diferenças finitas 
            result = np.array([
                self._h_finite_diff(float(u[i]), float(v[i]), deriv)
                for i in range(len(u))
            ])

        result = np.clip(result, eps, 1 - eps)
        return float(result[0]) if scalar_input else result

    def _h_finite_diff(self, u: float, v: float, deriv: int) -> float:
        eps = 1e-10
        if deriv == 1:
            delta = max(1e-6, min(v, 1 - v) * 1e-4)
            v_plus = min(v + delta, 1 - eps)
            v_minus = max(v - delta, eps)
            return np.clip(
                (self.cdf(u, v_plus) - self.cdf(u, v_minus)) / (v_plus - v_minus),
                eps, 1 - eps
            )
        else:
            delta = max(1e-6, min(u, 1 - u) * 1e-4)
            u_plus = min(u + delta, 1 - eps)
            u_minus = max(u - delta, eps)
            return np.clip(
                (self.cdf(u_plus, v) - self.cdf(u_minus, v)) / (u_plus - u_minus),
                eps, 1 - eps
            )

    def h_function_inv(self, h, v, deriv: int = 1,
                       max_iter: int = 200, tol: float = 1e-10):
        """Inversão da h-function"""
        eps = 1e-12
        scalar_input = np.ndim(h) == 0 and np.ndim(v) == 0
        h = np.atleast_1d(np.asarray(h, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        h = np.clip(h, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)

        # Inversão analítica para Gaussiana
        if self.family == 'gaussian' and deriv == 1:
            z_v = stats.norm.ppf(v)
            rho = self.param
            z_h = stats.norm.ppf(h)
            z_u = rho * z_v + np.sqrt(max(1 - rho ** 2, eps)) * z_h
            result = np.clip(stats.norm.cdf(z_u), eps, 1 - eps)
            return float(result[0]) if scalar_input else result

        # Bisseção vetorizada
        u_low = np.full_like(h, eps)
        u_high = np.full_like(h, 1 - eps)

        for _ in range(max_iter):
            u_mid = (u_low + u_high) / 2
            f_mid = self.h_function(u_mid, v, deriv=deriv) - h
            converged = (np.abs(f_mid) < tol) | ((u_high - u_low) < tol)
            if np.all(converged):
                break
            f_low = self.h_function(u_low, v, deriv=deriv) - h
            go_high = (f_mid * f_low) >= 0
            u_low = np.where(go_high, u_mid, u_low)
            u_high = np.where(go_high, u_high, u_mid)

        result = np.clip((u_low + u_high) / 2, eps, 1 - eps)
        return float(result[0]) if scalar_input else result

    def cdf(self, u, v):
        if self.param is None:
            raise ValueError("copula não ajustada")
        u = np.clip(u, 1e-10, 1 - 1e-10)
        v = np.clip(v, 1e-10, 1 - 1e-10)

        if self.family == 'gaussian':
            z_u = stats.norm.ppf(u)
            z_v = stats.norm.ppf(v)
            return stats.multivariate_normal.cdf(
                [z_u, z_v], mean=[0, 0],
                cov=[[1, self.param], [self.param, 1]]
            )
        elif self.family == 'clayton':
            theta = self.param
            return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
        elif self.family == 'gumbel':
            theta = self.param
            return np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))
        elif self.family == 'frank':
            theta = self.param
            if abs(theta) < 1e-10:
                return u * v
            return -1 / theta * np.log(
                1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)
            )
        elif self.family == 't':
            z_u = stats.t.ppf(u, self.param2)
            z_v = stats.t.ppf(v, self.param2)
            rho = self.param
            try:
                from scipy.integrate import dblquad
                nu = self.param2
                R = np.array([[1, rho], [rho, 1]])
                inv_R = np.linalg.inv(R)
                det = np.linalg.det(R)
                from scipy.special import gamma as gammaf
                def integrand(s, t):
                    z = np.array([s, t])
                    c = gammaf((nu + 2) / 2) / (gammaf(nu / 2) * nu * np.pi * np.sqrt(det))
                    return c * (1 + z @ inv_R @ z / nu) ** (-(nu + 2) / 2)
                result, _ = dblquad(integrand, -np.inf, z_u)
                return result
            except:
                return stats.multivariate_normal.cdf(
                    [z_u, z_v], mean=[0, 0], cov=[[1, rho], [rho, 1]]
                )
        else:
            raise NotImplementedError(f"CDF não implementada para {self.family}")

    def pdf(self, u, v):
        eps = 1e-10
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        if self.param is None:
            raise ValueError("cópula não ajustada")

        if self.family == 'gaussian':
            rho = self.param
            z_u = stats.norm.ppf(u)
            z_v = stats.norm.ppf(v)
            return (1.0 / np.sqrt(1 - rho ** 2)) * np.exp(
                -0.5 * (z_u ** 2 + z_v ** 2 - 2 * rho * z_u * z_v) / (1 - rho ** 2)
                + 0.5 * (z_u ** 2 + z_v ** 2)
            )
        elif self.family == 'clayton':
            theta = self.param
            if theta < eps:
                return 1.0
            term = u ** (-theta) + v ** (-theta) - 1
            if term <= 0:
                return eps
            return max((1 + theta) * (u * v) ** (-theta - 1) * term ** (-2 - 1 / theta), eps)
        elif self.family == 'gumbel':
            theta = self.param
            log_u = np.log(u)
            log_v = np.log(v)
            a = (-log_u) ** theta
            b = (-log_v) ** theta
            sum_ab = a + b
            if sum_ab <= eps:
                return eps
            sum_pow = sum_ab ** (1 / theta)
            C = np.exp(-sum_pow)
            density = C / (u * v) * sum_pow ** (2 - theta) * ((-log_u) * (-log_v)) ** (theta - 1) * (1 + (theta - 1) / sum_pow)
            return max(density, eps)
        elif self.family == 'frank':
            theta = self.param
            if abs(theta) < eps:
                return 1.0
            exp_theta = np.exp(-theta)
            exp_u = np.exp(-theta * u)
            exp_v = np.exp(-theta * v)
            num = -theta * (exp_theta - 1) * exp_u * exp_v
            den = ((exp_theta - 1) + (exp_u - 1) * (exp_v - 1)) ** 2
            return max(abs(num / den) if abs(den) > eps else eps, eps)
        elif self.family == 't':
            rho = self.param
            nu = self.param2
            z_u = stats.t.ppf(u, nu)
            z_v = stats.t.ppf(v, nu)
            from scipy.special import gamma
            R = np.array([[1, rho], [rho, 1]])
            det_R = 1 - rho ** 2
            z = np.array([z_u, z_v])
            qf = z @ np.linalg.inv(R) @ z
            constant = gamma((nu + 2) / 2) / (gamma(nu / 2) * nu * np.pi * np.sqrt(det_R))
            f_uv = constant * (1 + qf / nu) ** (-(nu + 2) / 2)
            return max(f_uv / (stats.t.pdf(z_u, nu) * stats.t.pdf(z_v, nu)), eps)
        else:
            raise NotImplementedError(f"PDF não implementada para {self.family}")

    def simulate_conditional(self, v, seed=None):
        if seed is not None:
            np.random.seed(seed)
        w = np.random.uniform(0, 1)
        return self.h_function_inv(w, v)

    def tail_dependence(self):
        if self.family == 'clayton':
            return (2 ** (-1 / self.param) if self.param > 0 else 0), 0
        elif self.family == 'gumbel':
            return 0, 2 - 2 ** (1 / self.param)
        elif self.family in ['gaussian', 'frank']:
            return 0, 0
        elif self.family == 't':
            rho, nu = self.param, self.param2
            lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
            return lam, lam
        return 0, 0

# CVineCopula — com max_trees e min_tau

class CVineCopula:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim
        self.trees = []
        self.copulas = {}
        self.order = None
        self.n_trees_fit = None 

    def fit(
        self,
        data: np.ndarray,
        families: Optional[List[str]] = None,
        auto_select: bool = True,
        order_method: str = 'dissmann',
        max_trees: Optional[int] = None,
        min_tau: float = 0.0,
    ) -> Dict:
        """
        Parâmetros
        max_trees : int, opcional
            Trunca a vine após `max_trees` árvores
        min_tau : float
            Pares com |Kendall tau| < min_tau são ajustados como independência
            (gaussian θ=0) sem tentar as demais famílias
        """
        if families is None:
            families = ['gaussian', 'clayton', 'gumbel', 'frank', 't']

        # Número máximo de árvores: por padrão n_dim-1, truncado se max_trees fornecido
        n_trees = self.n_dim - 1
        if max_trees is not None:
            n_trees = min(max_trees, n_trees)
        self.n_trees_fit = n_trees

        logger.info(f"Ajustando C-vine com {self.n_dim} variaveis")
        logger.info(f"  árvores: {n_trees}/{self.n_dim - 1} | min_tau={min_tau}")

        if order_method == 'dissmann':
            self._select_order(data)
        elif order_method == 'mutual_info':
            self._select_order_mutual_info(data)
        elif order_method == 'sum':
            self._select_order_simple(data)
        else:
            raise ValueError(f"método desconhecido: {order_method}")

        data_ordered = data[:, self.order]
        pseudo_data = [data_ordered.copy()]

        for tree_level in range(n_trees):  # <- truncado aqui
            logger.info(f"árvore {tree_level + 1}/{n_trees}")

            current_data = pseudo_data[tree_level]
            n_edges = self.n_dim - tree_level - 1
            next_level_data = []

            for edge in range(n_edges):
                var1 = 0 if tree_level > 0 else tree_level
                var2 = edge + 1 if tree_level > 0 else tree_level + edge + 1

                u = current_data[:, var1]
                v = current_data[:, var2]

                # pula ajuste para pares sem dependência
                tau = abs(PairCopula.kendall_tau(u, v))
                if tau < min_tau:
                    best_family = 'gaussian'
                    best_copula = PairCopula('gaussian')
                    best_copula.param = 0.0
                elif auto_select:
                    best_family, best_copula = self._select_best_copula(u, v, families)
                else:
                    best_family = families[0]
                    best_copula = PairCopula(best_family)
                    best_copula.fit(u, v)

                key = (tree_level, var1, var2)
                self.copulas[key] = best_copula
                logger.info(f"Edge ({var1},{var2}): {best_family}, θ={best_copula.param:.3f}")

                # Calcula h-functions vetorizadas para próximo nível
                if tree_level < n_trees - 1:
                    h_values = best_copula.h_function(v, u, deriv=1)  
                    next_level_data.append(h_values)

            if tree_level < n_trees - 1 and next_level_data:
                pseudo_data.append(np.column_stack(next_level_data))

        self.trees = pseudo_data
        logger.info("C-vine ajustada com sucesso")

        return {
            'n_trees': n_trees,
            'order': self.order,
            'copulas': self.copulas
        }

    def _select_order(self, data: np.ndarray):
        n = data.shape[1]
        tau_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    tau_matrix[i, j] = abs(PairCopula.kendall_tau(data[:, i], data[:, j]))

        selected, remaining = [], list(range(n))
        tau_sums = np.sum(tau_matrix, axis=1)
        root = np.argmax(tau_sums)
        selected.append(root)
        remaining.remove(root)
        logger.info(f"variavel raíz selecionada: {root} (tau_sum={tau_sums[root]:.3f}")

        while remaining:
            best_score, best_var = -np.inf, remaining[0]
            for candidate in remaining:
                score = sum(
                    (1.0 / (len(selected) - idx)) * tau_matrix[candidate, sel]
                    for idx, sel in enumerate(selected)
                )
                if score > best_score:
                    best_score, best_var = score, candidate
            selected.append(best_var)
            remaining.remove(best_var)
            logger.info(f"adicionada var {best_var} (score={best_score:.3f})")

        self.order = np.array(selected)
        logger.info(f"ordem final selecionada: {self.order}")

    def _select_order_mutual_info(self, data: np.ndarray):
        from sklearn.feature_selection import mutual_info_regression
        n = data.shape[1]
        mi_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    mi_matrix[i, j] = mutual_info_regression(
                        data[:, i].reshape(-1, 1), data[:, j], random_state=42
                    )[0]
        selected, remaining = [], list(range(n))
        mi_sums = np.sum(mi_matrix, axis=1)
        root = np.argmax(mi_sums)
        selected.append(root)
        remaining.remove(root)
        logger.info(f"raiz (MI): {root} (MI_sum={mi_sums[root]:.3f})")
        while remaining:
            best_mi, best_var = -np.inf, remaining[0]
            for candidate in remaining:
                mi_score = np.mean([mi_matrix[candidate, sel] for sel in selected])
                if mi_score > best_mi:
                    best_mi, best_var = mi_score, candidate
            selected.append(best_var)
            remaining.remove(best_var)
            logger.info(f"adicionada: {best_var} (MI={best_mi:.3f})")
        self.order = np.array(selected)
        logger.info(f"ordem MI: {self.order}")

    def _select_order_simple(self, data: np.ndarray):
        n = data.shape[1]
        tau_sums = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    tau_sums[i] += abs(PairCopula.kendall_tau(data[:, i], data[:, j]))
        self.order = np.argsort(-tau_sums)
        logger.info(f"ordem selecionada (simples): {self.order}")

    def _select_best_copula(self, u, v, families):
        best_aic, best_family, best_copula = np.inf, None, None
        for family in families:
            try:
                copula = PairCopula(family)
                copula.fit(u, v)
                ll = self._loglikelihood(u, v, copula)
                k = 2 if family != 't' else 3
                aic = 2 * k - 2 * ll
                if aic < best_aic:
                    best_aic, best_family, best_copula = aic, family, copula
            except Exception as e:
                logger.warning(f"falha ao ajustar {family}: {e}")
                continue

        if best_copula is None:
            logger.warning("Todas familias falharam, usando independencia")
            best_family = 'gaussian'
            best_copula = PairCopula('gaussian')
            best_copula.param = 0.0

        return best_family, best_copula

    def _loglikelihood(self, u, v, copula):
        eps = 1e-10
        ll = 0
        for i in range(len(u)):
            ui = np.clip(u[i], eps, 1 - eps)
            vi = np.clip(v[i], eps, 1 - eps)
            try:
                density = max(copula.pdf(ui, vi), eps)
                ll += np.log(density)
            except:
                ll += np.log(eps)
        return ll

    def simulate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulação C-vine
        """
        if seed is not None:
            np.random.seed(seed)

        n = n_samples
        d = self.n_dim
        n_trees = self.n_trees_fit if self.n_trees_fit is not None else d - 1

        W = np.random.uniform(0, 1, (n, d))
        V = np.zeros((n, d))
        V[:, 0] = W[:, 0]

        h = np.zeros((n, d, d))
        for k in range(d):
            h[:, 0, k] = V[:, 0]

        for k in range(1, d):
            w = W[:, k].copy() 

            for tree in range(min(k, n_trees) - 1, -1, -1):
                key = (tree, 0, k - tree)
                if key not in self.copulas:
                    continue
                copula = self.copulas[key]

                # Condicionante 
                v_cond = V[:, 0] if tree == 0 else h[:, 0, tree - 1]

                # h_function_inv 
                w = copula.h_function_inv(w, v_cond, deriv=1)

            V[:, k] = w

            # Calcula h-functions para próximas iterações 
            if k < d - 1:
                key_0 = (0, 0, k)
                if key_0 in self.copulas:
                    h[:, k, 0] = self.copulas[key_0].h_function(V[:, k], V[:, 0], deriv=1)
                else:
                    h[:, k, 0] = V[:, k]

                for tree in range(1, min(k, n_trees)):
                    key = (tree, 0, k - tree)
                    if key in self.copulas:
                        copula = self.copulas[key]
                        u_val = h[:, k - tree, tree - 1]
                        v_val = h[:, 0, tree - 1]
                        h[:, k - tree, tree] = copula.h_function(u_val, v_val, deriv=1)
                    else:
                        h[:, k - tree, tree] = h[:, k - tree, tree - 1]

        # Reverte para ordem original
        samples_original = np.zeros_like(V)
        for i, orig_idx in enumerate(self.order):
            samples_original[:, orig_idx] = V[:, i]

        return samples_original


# DVineCopula 

class DVineCopula:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim
        self.copulas = {}
        self.order = None

    def fit(
        self,
        data: np.ndarray,
        families: Optional[List[str]] = None,
        max_trees: Optional[int] = None,
        min_tau: float = 0.0,
    ) -> Dict:
        if families is None:
            families = ['gaussian', 'clayton', 'gumbel']

        n_trees = self.n_dim - 1
        if max_trees is not None:
            n_trees = min(max_trees, n_trees)

        logger.info(f"ajustamos D-vine com {self.n_dim} variaveis")
        self._select_sequential_order(data)
        data_ordered = data[:, self.order]
        pseudo_data = [data_ordered.copy()]

        for tree_level in range(n_trees):
            logger.info(f"árvore {tree_level + 1}/{n_trees}")
            current_data = pseudo_data[tree_level]
            n_pairs = current_data.shape[1] - tree_level - 1
            next_level_data = []

            for i in range(n_pairs):
                u = current_data[:, i]
                v = current_data[:, i + 1]

                tau = abs(PairCopula.kendall_tau(u, v))
                if tau < min_tau:
                    copula = PairCopula('gaussian')
                    copula.param = 0.0
                else:
                    copula = PairCopula(families[0])
                    copula.fit(u, v)

                self.copulas[(tree_level, i, i + 1)] = copula
                logger.info(f"Pair ({i}, {i+1}): θ={copula.param:.3f}")

                if tree_level < n_trees - 1:
                    h_values = copula.h_function(v, u, deriv=1)  
                    next_level_data.append(h_values)

            if next_level_data:
                pseudo_data.append(np.column_stack(next_level_data))

        logger.info("D-vine ajustada")
        return {'n_trees': n_trees, 'copulas': self.copulas}

    def _select_sequential_order(self, data: np.ndarray):
        n = data.shape[1]
        max_tau, best_pair = -1, (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                tau = abs(PairCopula.kendall_tau(data[:, i], data[:, j]))
                if tau > max_tau:
                    max_tau, best_pair = tau, (i, j)
        used = [best_pair[0], best_pair[1]]
        remaining = [i for i in range(n) if i not in used]
        while remaining:
            max_tau, best_next = -1, remaining[0]
            for candidate in remaining:
                tau = abs(PairCopula.kendall_tau(data[:, used[-1]], data[:, candidate]))
                if tau > max_tau:
                    max_tau, best_next = tau, candidate
            used.append(best_next)
            remaining.remove(best_next)
        self.order = np.array(used)
        logger.info(f"ordem D-Vine: {self.order}")

    def simulate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        samples = np.zeros((n_samples, self.n_dim))
        samples[:, 0] = np.random.uniform(0, 1, n_samples)
        for i in range(1, self.n_dim):
            key = (0, i - 1, i)
            if key in self.copulas:
                copula = self.copulas[key]
                for j in range(n_samples):
                    samples[j, i] = copula.simulate_conditional(samples[j, i - 1])
            else:
                samples[:, i] = np.random.uniform(0, 1, n_samples)
        samples_original = np.zeros_like(samples)
        for i, orig_idx in enumerate(self.order):
            samples_original[:, orig_idx] = samples[:, i]
        return samples_original