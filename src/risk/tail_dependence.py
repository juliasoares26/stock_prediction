import pandas as pd 
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TailDependence:
    """
    λ_L (lower): P(Y < F_Y^{-1}(u) | X < F_X^{-1}(u)) quando u 0
    λ_U (upper): P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u)) quando u→1
    """

    def __init__(self):
        self.data = None
        self.asset_names = None
        self.lambda_L_matrix = None
        self.lambda_U_matrix = None
        self.best_copulas = {}

    def fit(self, returns: pd.DataFrame, method: str = 'empirical'):
        self.data = returns
        self.asset_names = returns.columns.tolist()
        n_assets = len(self.asset_names)

        logger.info(f"calculando tail dependence: {n_assets} ativos, método={method}")

        # Matrizes de dependencias
        self.lambda_L_matrix = np.zeros((n_assets, n_assets))
        self.lambda_U_matrix = np.zeros((n_assets, n_assets))

        for i in range(n_assets):
            for j in range(i, n_assets):
                if i == j:
                    self.lambda_L_matrix[i, j] = 1.0
                    self.lambda_U_matrix[i, j] = 1.0
                else: 
                    x = returns.iloc[:, i].values
                    y = returns.iloc[:, j].values

                    if method == 'empirical':
                        lambda_L, lambda_U = self._empirical_tail_dependence(x, y)
                    elif method == 'parametric':
                        lambda_L, lambda_U, best_copula = self._parametric_tail_dependence(x, y)
                        self.best_copulas[(self.asset_names[i], self.asset_names[j])] = best_copula
                    elif method == 'hill':
                        lambda_L, lambda_U = self._hill_tail_dependence(x, y)
                    else:
                        raise ValueError(f"Método desconhecido: {method}")
                    
                    # atribuições corretas com simetria
                    self.lambda_L_matrix[i, j] = lambda_L
                    self.lambda_L_matrix[j, i] = lambda_L

                    self.lambda_U_matrix[i, j] = lambda_U
                    self.lambda_U_matrix[j, i] = lambda_U

        logger.info(f"tail dependence calculado")
        logger.info(f" λ_L médio: {self._mean_off_diagonal(self.lambda_L_matrix):.4f}")
        logger.info(f" λ_U médio: {self._mean_off_diagonal(self.lambda_U_matrix):.4f}")
        
        return {
            'lambda_L': self.lambda_L_matrix,
            'lambda_U': self.lambda_U_matrix
        }
    
    def _empirical_tail_dependence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.05
    ) -> Tuple[float, float]:
        """Estimador empírico"""
        n = len(x)
        u = stats.rankdata(x) / (n + 1)
        v = stats.rankdata(y) / (n + 1)

        # Lower tail
        lower_threshold = threshold
        both_lower = np.sum((u <= lower_threshold) & (v <= lower_threshold))
        x_lower = np.sum(u <= lower_threshold)

        if x_lower > 0:
            lambda_L = both_lower / x_lower  
        else:
            lambda_L = 0.0

        # Upper tail
        upper_threshold = 1 - threshold
        both_upper = np.sum((u >= upper_threshold) & (v >= upper_threshold))
        x_upper = np.sum(u >= upper_threshold)

        if x_upper > 0:
            lambda_U = both_upper / x_upper  
        else:
            lambda_U = 0.0

        return lambda_L, lambda_U
    
    def _parametric_tail_dependence(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, str]:
        """Estimação paramétrica via seleção de modelo"""
        n = len(x)

        # Pseudo-obs uniformes
        u = stats.rankdata(x) / (n + 1)
        v = stats.rankdata(y) / (n + 1)

        # Ajustando famílias diferentes de copulas
        copulas_to_fit = [
            'gaussian',
            't-student',
            'clayton',  
            'gumbel',
            'rotated_clayton',  
            'joe'
        ]

        results = {}

        for copula_name in copulas_to_fit:
            try:
                params, loglik, aic = self._fit_copula(u, v, copula_name)
                lambda_L, lambda_U = self._theoretical_tail_dependence(copula_name, params)

                results[copula_name] = {
                    'params': params,
                    'loglik': loglik,
                    'aic': aic,
                    'lambda_L': lambda_L,
                    'lambda_U': lambda_U
                }
            except Exception as e:
                logger.warning(f"falha ao ajustar {copula_name}: {e}")
                continue

        # Este bloco estava dentro do for loop
        if not results:
            logger.warning("nenhuma copula ajustada, usando empírico")
            lambda_L, lambda_U = self._empirical_tail_dependence(x, y)
            return lambda_L, lambda_U, 'empirical'
        
        best_copula = min(results.keys(), key=lambda k: results[k]['aic'])

        # Extrai λ da melhor copula
        lambda_L = results[best_copula]['lambda_L']
        lambda_U = results[best_copula]['lambda_U']

        logger.debug(f"Melhor copula: {best_copula} (AIC={results[best_copula]['aic']:.2f})")
        return lambda_L, lambda_U, best_copula
    
    def _fit_copula(
        self,
        u: np.ndarray,
        v: np.ndarray,
        copula_name: str
    ) -> Tuple[Dict, float, float]:
        """Copula via maxima verossimilhança"""
        n = len(u)

        if copula_name == 'gaussian':
            # ρ ∈ (-1, 1)
            def neg_loglik(params):
                rho = params[0]
                rho = np.clip(rho, -0.999, 0.999)

                z_u = stats.norm.ppf(u)
                z_v = stats.norm.ppf(v)

                loglik = -0.5 * np.log(1 - rho**2) - (rho**2 * (z_u**2 + z_v**2) - 2*rho*z_u*z_v) / (2*(1-rho**2))

                return -np.sum(loglik)

            from scipy.stats import kendalltau
            tau, _ = kendalltau(u, v)
            rho_init = np.sin(np.pi * tau / 2)
            
            result = minimize(neg_loglik, [rho_init], bounds=[(-0.999, 0.999)], method='L-BFGS-B')

            params = {'rho': result.x[0]}
            loglik = -result.fun
            k = 1
            aic = 2 * k - 2 * loglik

        elif copula_name == 't-student':
            # ρ, ν - graus de liberdade
            def neg_loglik(params):
                rho, nu = params
                rho = np.clip(rho, -0.999, 0.999)
                nu = max(nu, 2.01)

                t_u = stats.t.ppf(u, nu)
                t_v = stats.t.ppf(v, nu)

                det_R = 1 - rho**2

                quad_form = (t_u**2 + t_v**2 - 2*rho*t_u*t_v) / det_R

                from scipy.special import gammaln

                loglik = (
                    gammaln((nu+2)/2) - gammaln(nu/2) - 0.5*np.log(det_R) +
                    gammaln(nu/2) - 2*gammaln((nu+1)/2) -
                    ((nu+2)/2) * np.log(1 + quad_form/nu) +
                    ((nu+1)/2) * np.log(1 + t_u**2/nu) +
                    ((nu+1)/2) * np.log(1 + t_v**2/nu)
                )

                return -np.sum(loglik)
            
            from scipy.stats import kendalltau
            tau, _ = kendalltau(u, v)
            rho_init = np.sin(np.pi * tau / 2)

            result = minimize(neg_loglik, [rho_init, 5.0],
                            bounds=[(-0.999, 0.999), (2.01, 30)],
                            method='L-BFGS-B')
            
            params = {'rho': result.x[0], 'nu': result.x[1]}
            loglik = -result.fun
            k = 2 
            aic = 2 * k - 2 * loglik

        elif copula_name == 'clayton':
            def neg_loglik(params):
                theta = max(params[0], 0.01)
                loglik = np.log(1 + theta) - (1+theta) * (np.log(u) + np.log(v)) - (1/theta + 2) * np.log(u**(-theta) + v**(-theta) - 1)
                return -np.sum(loglik)
            
            from scipy.stats import kendalltau
            tau, _ = kendalltau(u, v)
            theta_init = max(2 * tau / (1 - tau), 0.01)
            
            result = minimize(neg_loglik, [theta_init], bounds=[(0.01, 20)], method='L-BFGS-B')
            
            params = {'theta': result.x[0]}
            loglik = -result.fun
            k = 1
            aic = 2 * k - 2 * loglik
                
        elif copula_name == 'gumbel':
            def neg_loglik(params):
                theta = max(params[0], 1.01)
                
                A = (-np.log(u))**theta + (-np.log(v))**theta
                C = np.exp(-A**(1/theta))
                
                loglik = np.log(C) + np.log(A**(1/theta - 2)) + np.log(theta - 1 + A**(1/theta)) - (theta - 1) * (np.log(u) + np.log(v))
                
                return -np.sum(loglik)
            
            from scipy.stats import kendalltau
            tau, _ = kendalltau(u, v)
            theta_init = max(1 / (1 - tau), 1.01)
            
            result = minimize(neg_loglik, [theta_init], bounds=[(1.01, 20)], method='L-BFGS-B')
            
            params = {'theta': result.x[0]}
            loglik = -result.fun
            k = 1
            aic = 2 * k - 2 * loglik
            
        elif copula_name == 'rotated_clayton':
            u_rot = 1 - u
            v_rot = 1 - v
            
            def neg_loglik(params):
                theta = max(params[0], 0.01)
                loglik = np.log(1 + theta) - (1+theta) * (np.log(u_rot) + np.log(v_rot)) - (1/theta + 2) * np.log(u_rot**(-theta) + v_rot**(-theta) - 1)
                return -np.sum(loglik)
            
            from scipy.stats import kendalltau
            tau, _ = kendalltau(u_rot, v_rot)
            theta_init = max(2 * tau / (1 - tau), 0.01)
            
            result = minimize(neg_loglik, [theta_init], bounds=[(0.01, 20)], method='L-BFGS-B')
            
            params = {'theta': result.x[0]}
            loglik = -result.fun
            k = 1
            aic = 2 * k - 2 * loglik
            
        elif copula_name == 'joe':
            def neg_loglik(params):
                theta = max(params[0], 1.01)
                
                # Protege contra valores zero
                term1 = np.clip(1 - (1-u)**theta, 1e-10, 1-1e-10)
                term2 = np.clip(1 - (1-v)**theta, 1e-10, 1-1e-10)
                
                # Densidade Joe
                with np.errstate(invalid='ignore'): 
                    C = 1 - (term1**(-1/theta) + term2**(-1/theta) - (term1 * term2)**(-1/theta))**theta
                
                C = np.clip(C, 1e-10, 1-1e-10)
                
                loglik = np.log(C + 1e-10)
                
                return -np.sum(loglik)
            
            result = minimize(neg_loglik, [2.0], bounds=[(1.01, 10)], method='L-BFGS-B')
            
            params = {'theta': result.x[0]}
            loglik = -result.fun
            k = 1
            aic = 2 * k - 2 * loglik
            
        else:
            raise ValueError(f"Copula desconhecida: {copula_name}")
        
        return params, loglik, aic
    
    def _theoretical_tail_dependence(
        self,
        copula_name: str,
        params: Dict
    ) -> Tuple[float, float]:
        """Calcula λ_L e λ_U teóricos para copula específica"""
        if copula_name == 'gaussian':
            return 0.0, 0.0
        
        elif copula_name == 't-student':
            rho = params['rho']
            nu = params['nu']
            
            if abs(rho) < 0.999:
                arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                lambda_td = 2 * stats.t.cdf(arg, nu + 1)
                return lambda_td, lambda_td
            else:
                return 0.0, 0.0
        
        elif copula_name == 'clayton':
            theta = params['theta']
            if theta > 0:
                lambda_L = 2 ** (-1 / theta)
                return lambda_L, 0.0
            else:
                return 0.0, 0.0
        
        elif copula_name == 'gumbel':
            theta = params['theta']
            if theta >= 1:
                lambda_U = 2 - 2 ** (1 / theta)
                return 0.0, lambda_U
            else:
                return 0.0, 0.0
        
        elif copula_name == 'rotated_clayton':
            theta = params['theta']
            if theta > 0:
                lambda_U = 2 ** (-1 / theta)
                return 0.0, lambda_U
            else:
                return 0.0, 0.0
        
        elif copula_name == 'joe':
            theta = params['theta']
            if theta >= 1:
                lambda_U = 2 - 2 ** (1 / theta)
                return 0.0, lambda_U
            else:
                return 0.0, 0.0
        
        else:
            return 0.0, 0.0
    
    def _hill_tail_dependence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[float, float]:
        """Estimador de Hill para tail dependence"""
        n = len(x)
        
        if k is None:
            k = int(np.sqrt(n))
        
        sorted_idx_x = np.argsort(x)
        sorted_idx_y = np.argsort(y)
        
        lower_x = set(sorted_idx_x[:k])
        lower_y = set(sorted_idx_y[:k])
        both_lower = len(lower_x & lower_y)
        
        lambda_L = both_lower / k if k > 0 else 0
        
        upper_x = set(sorted_idx_x[-k:])
        upper_y = set(sorted_idx_y[-k:])
        both_upper = len(upper_x & upper_y)
        
        lambda_U = both_upper / k if k > 0 else 0
        
        return lambda_L, lambda_U
    
    @staticmethod
    def _mean_off_diagonal(matrix: np.ndarray) -> float:
        """Média dos elementos fora da diagonal"""
        n = matrix.shape[0]
        off_diag = matrix[np.triu_indices(n, k=1)]
        return np.mean(off_diag)
    
    def get_lambda_L_dataframe(self) -> pd.DataFrame:
        """Retorna matriz λ_L como DataFrame"""
        return pd.DataFrame(
            self.lambda_L_matrix,
            index=self.asset_names,
            columns=self.asset_names
        )
    
    def get_lambda_U_dataframe(self) -> pd.DataFrame:
        """Retorna matriz λ_U como DataFrame"""
        return pd.DataFrame(
            self.lambda_U_matrix,
            index=self.asset_names,
            columns=self.asset_names
        )
    
    def get_best_copulas_summary(self) -> pd.DataFrame:
        """Retorna sumário das melhores copulas selecionadas"""
        if not self.best_copulas:
            return pd.DataFrame()
        
        summary = []
        for (asset1, asset2), copula in self.best_copulas.items():
            idx1 = self.asset_names.index(asset1)
            idx2 = self.asset_names.index(asset2)
            
            summary.append({
                'Asset_1': asset1,
                'Asset_2': asset2,
                'Best_Copula': copula,
                'Lambda_L': self.lambda_L_matrix[idx1, idx2],
                'Lambda_U': self.lambda_U_matrix[idx1, idx2]
            })
        
        return pd.DataFrame(summary)
    
    def test_tail_independence(
        self,
        asset1: str,
        asset2: str,
        threshold: float = 0.05,
        n_bootstrap: int = 1000
    ) -> Dict:
        """Testa H0: λ = 0 (tail independence) via bootstrap"""
        logger.info(f"Testando tail independence: {asset1} vs {asset2}")
        
        x = self.data[asset1].values
        y = self.data[asset2].values
        
        lambda_L_obs, lambda_U_obs = self._empirical_tail_dependence(x, y, threshold)
        
        lambda_L_boot = []
        lambda_U_boot = []
        
        n = len(x)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            x_boot = x[idx]
            y_boot = y[idx]
            
            lambda_L_b, lambda_U_b = self._empirical_tail_dependence(x_boot, y_boot, threshold)
            lambda_L_boot.append(lambda_L_b)
            lambda_U_boot.append(lambda_U_b)
        
        lambda_L_boot = np.array(lambda_L_boot)
        lambda_U_boot = np.array(lambda_U_boot)
        
        p_value_L = np.mean(lambda_L_boot == 0)
        p_value_U = np.mean(lambda_U_boot == 0)
        
        logger.info(f"  λ_L = {lambda_L_obs:.4f}, p-value = {p_value_L:.4f}")
        logger.info(f"  λ_U = {lambda_U_obs:.4f}, p-value = {p_value_U:.4f}")
        
        return {
            'lambda_L': lambda_L_obs,
            'lambda_U': lambda_U_obs,
            'pvalue_L': p_value_L,
            'pvalue_U': p_value_U,
            'reject_independence_L': p_value_L < 0.05,
            'reject_independence_U': p_value_U < 0.05
        }
    
    def tail_network(self, threshold_dependence: float = 0.1) -> Dict:
        """Constrói rede de tail dependence"""
        logger.info(f"Construindo tail network (threshold={threshold_dependence})")
        
        n_assets = len(self.asset_names)
        edges_lower = []
        edges_upper = []
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                lambda_L = self.lambda_L_matrix[i, j]
                lambda_U = self.lambda_U_matrix[i, j]
                
                if lambda_L > threshold_dependence:
                    edges_lower.append((self.asset_names[i], self.asset_names[j], lambda_L))
                
                if lambda_U > threshold_dependence:
                    edges_upper.append((self.asset_names[i], self.asset_names[j], lambda_U))
        
        degree_lower = {asset: 0 for asset in self.asset_names}
        degree_upper = {asset: 0 for asset in self.asset_names}
        
        for asset1, asset2, _ in edges_lower:
            degree_lower[asset1] += 1
            degree_lower[asset2] += 1
        
        for asset1, asset2, _ in edges_upper:
            degree_upper[asset1] += 1
            degree_upper[asset2] += 1
        
        logger.info(f"Lower tail: {len(edges_lower)} edges")
        logger.info(f"Upper tail: {len(edges_upper)} edges")
        
        return {
            'edges_lower': edges_lower,
            'edges_upper': edges_upper,
            'degree_lower': degree_lower,
            'degree_upper': degree_upper
        }
    
    def quantile_dependence(
        self,
        asset1: str,
        asset2: str,
        quantiles: np.ndarray = np.linspace(0.01, 0.99, 99)
    ) -> pd.DataFrame:
        """Dependência condicional ao longo de quantis"""
        x = self.data[asset1].values
        y = self.data[asset2].values
        n = len(x)
        
        u = stats.rankdata(x) / (n + 1)
        v = stats.rankdata(y) / (n + 1)
        
        results = []
        
        for q in quantiles:
            x_in_quantile = (u <= q) & (u > q - 0.05)
            
            if x_in_quantile.sum() > 10:
                y_given_x = v[x_in_quantile]
                mean_y_given_x = y_given_x.mean()
                
                results.append({
                    'quantile': q,
                    'conditional_mean': mean_y_given_x,
                    'conditional_std': y_given_x.std()
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    np.random.seed(42)
    
    from scipy.stats import multivariate_t
    
    n = 1000
    rho = 0.7
    nu = 5
    
    cov = np.array([[1, rho], [rho, 1]])
    data_t = multivariate_t.rvs(df=nu, shape=cov, size=n)
    
    returns = pd.DataFrame(data_t, columns=['Asset_A', 'Asset_B'])
    
    td = TailDependence()
    td.fit(returns, method='parametric')
    
    print(f"λ_L: {td.lambda_L_matrix[0, 1]:.4f}")
    print(f"λ_U: {td.lambda_U_matrix[0, 1]:.4f}")
    
    print(f"\nMelhor copula: {td.best_copulas[('Asset_A', 'Asset_B')]}")
    
    print("\n", td.get_best_copulas_summary())
    
    test_result = td.test_tail_independence('Asset_A', 'Asset_B')
    print(f"\nTeste de independência:")
    print(f"Rejeita H0 (lower): {test_result['reject_independence_L']}")
    print(f"Rejeita H0 (upper): {test_result['reject_independence_U']}")