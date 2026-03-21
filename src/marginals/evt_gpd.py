import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from scipy import stats, optimize
from scipy.special import gamma
import warnings
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPD:
    """
    Generalized Pareto Distribution para Extreme Value Theory
    """
    
    def __init__(self):
        self.xi = None
        self.sigma = None
        self.threshold = None
        self.n_exceedances = None
        self.exceedances = None
        self.method = None
        self.converged = None
        self.std_errors = None
        self.hessian = None
        self.upper_endpoint = None  # Para ξ < 0
    
    def select_threshold(
        self,
        data: np.ndarray,
        method: str = 'mrl',
        plot: bool = False,
        **kwargs
    ) -> float:
        """
        Seleção automática de threshold
        
        Métodos:
        mrl: Mean Residual Life Plot
        stability: Parameter stability plot
        automated: Automated selection (Danielsson et al.)
        """
        if method == 'mrl':
            return self._threshold_mrl(data, plot=plot, **kwargs)
        elif method == 'stability':
            return self._threshold_stability(data, plot=plot, **kwargs)
        elif method == 'automated':
            return self._threshold_automated(data, **kwargs)
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def _threshold_mrl(
        self,
        data: np.ndarray,
        n_thresholds: int = 50,
        min_exceedances: int = 50,
        plot: bool = False
    ) -> float:
        """Mean Residual Life Plot para seleção de threshold"""
        sorted_data = np.sort(data)
        n = len(data)
        
        # Range de thresholds possíveis
        max_idx = n - min_exceedances
        thresholds = sorted_data[np.linspace(0, max_idx, n_thresholds, dtype=int)]
        
        mrl_values = []
        for u in thresholds:
            exceedances = data[data > u] - u
            if len(exceedances) >= min_exceedances:
                mrl_values.append(np.mean(exceedances))
            else:
                mrl_values.append(np.nan)
        
        mrl_values = np.array(mrl_values)
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, mrl_values, 'b-', linewidth=2)
            plt.xlabel('Threshold')
            plt.ylabel('Mean Excess')
            plt.title('Mean Residual Life Plot')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Seleciona threshold onde MRL se estabiliza 
        # Usa o ponto onde a segunda derivada é mínima
        valid_idx = ~np.isnan(mrl_values)
        if np.sum(valid_idx) < 3:
            logger.warning("Poucos pontos válidos, usando percentil 90")
            return np.percentile(data, 90)
        
        valid_mrl = mrl_values[valid_idx]
        valid_thresh = thresholds[valid_idx]
        
        # Segunda derivada numérica
        if len(valid_mrl) > 2:
            second_deriv = np.gradient(np.gradient(valid_mrl))
            # Evita extremos
            mid_start = len(second_deriv) // 4
            mid_end = 3 * len(second_deriv) // 4
            optimal_idx = mid_start + np.argmin(np.abs(second_deriv[mid_start:mid_end]))
            return valid_thresh[optimal_idx]
        else:
            return np.percentile(data, 90)
    
    def _threshold_stability(
        self,
        data: np.ndarray,
        n_thresholds: int = 30,
        min_exceedances: int = 50,
        plot: bool = False
    ) -> float:
        """Parameter stability plot"""
        sorted_data = np.sort(data)
        n = len(data)
        
        max_idx = n - min_exceedances
        thresholds = sorted_data[np.linspace(0, max_idx, n_thresholds, dtype=int)]
        
        xi_vals = []
        sigma_vals = []
        
        for u in thresholds:
            try:
                temp_gpd = GPD()
                temp_gpd.fit(data, u, method='mle', return_std_errors=False)
                if temp_gpd.converged:
                    xi_vals.append(temp_gpd.xi)
                    sigma_vals.append(temp_gpd.sigma)
                else:
                    xi_vals.append(np.nan)
                    sigma_vals.append(np.nan)
            except:
                xi_vals.append(np.nan)
                sigma_vals.append(np.nan)
        
        xi_vals = np.array(xi_vals)
        sigma_vals = np.array(sigma_vals)
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            ax1.plot(thresholds, xi_vals, 'b-', linewidth=2)
            ax1.set_ylabel('Shape (ξ)')
            ax1.set_title('Parameter Stability Plot')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(thresholds, sigma_vals, 'r-', linewidth=2)
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Scale (σ)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Seleciona threshold onde ξ é mais estável
        valid_idx = ~np.isnan(xi_vals)
        if np.sum(valid_idx) < 3:
            return np.percentile(data, 90)
        
        valid_xi = xi_vals[valid_idx]
        valid_thresh = thresholds[valid_idx]
        
        # Usa variância móvel para encontrar região estável
        window = min(5, len(valid_xi) // 2)
        rolling_std = pd.Series(valid_xi).rolling(window=window, center=True).std()
        
        # Encontra região de menor variabilidade
        stable_idx = np.nanargmin(rolling_std)
        return valid_thresh[stable_idx]
    
    def _threshold_automated(
        self,
        data: np.ndarray,
        alpha: float = 0.05
    ) -> float:
        """
        Automated threshold selection baseado em Danielsson et al.
        Minimiza MSE assintótico
        """
        n = len(data)
        sorted_data = np.sort(data)
        
        # Range de k (número de exceedances)
        k_min = max(10, int(np.sqrt(n)))
        k_max = min(int(n * 0.5), n - 10)
        k_range = np.arange(k_min, k_max, max(1, (k_max - k_min) // 50))
        
        best_k = k_min
        min_amse = np.inf
        
        for k in k_range:
            threshold = sorted_data[n - k]
            try:
                temp_gpd = GPD()
                temp_gpd.fit(data, threshold, method='mle', return_std_errors=False)
                
                if temp_gpd.converged and temp_gpd.std_errors is not None:
                    # AMSE = bias² + variance
                    # Aproximação: variance ∝ 1/k, bias² ∝ k²/n²
                    variance_term = temp_gpd.std_errors[0]**2
                    bias_term = (k / n)**2
                    amse = variance_term + bias_term
                    
                    if amse < min_amse:
                        min_amse = amse
                        best_k = k
            except:
                continue
        
        optimal_threshold = sorted_data[n - best_k]
        logger.info(f"threshold automático: {optimal_threshold:.4f} (k={best_k})")
        return optimal_threshold
    
    def fit(
        self,
        data: np.ndarray,
        threshold: float,
        method: str = 'mle',
        return_std_errors: bool = True,
        use_analytical_hessian: bool = True
    ) -> Dict:
        """
        Ajusta GPD aos dados
        
        Argumentos:
        data: dados observados
        threshold: limiar para exceedances
        method: 'mle' ou 'pwm'
        return_std_errors: calcular erros padrão
        use_analytical_hessian: usar Hessiana analítica 
        """
        self.threshold = threshold
        
        # Extrai exceedances
        self.exceedances = data[data > threshold] - threshold
        self.n_exceedances = len(self.exceedances)
        
        if self.n_exceedances < 10:
            logger.warning(f"Apenas {self.n_exceedances} exceedances - resultados instáveis")
        
        logger.info(f"Fitting GPD: {self.n_exceedances}/{len(data)} exceedances (threshold={threshold:.4f})")
        
        if method == 'mle':
            self._fit_mle()
        elif method == 'pwm':
            self._fit_pwm()
        else:
            raise ValueError(f"Método desconhecido: {method}")
        
        self.method = method
        
        # Calcula upper endpoint para ξ < 0
        if self.xi < 0:
            self.upper_endpoint = self.threshold - self.sigma / self.xi
            logger.info(f"ξ < 0: upper endpoint = {self.upper_endpoint:.4f}")
        else:
            self.upper_endpoint = np.inf
        
        # Erros padrão
        if return_std_errors and self.converged and method == 'mle':
            if use_analytical_hessian:
                self._calculate_std_errors_analytical()
            else:
                self._calculate_std_errors_numerical()
        
        # Diagnósticos
        diagnostics = self._diagnostics()
        
        logger.info(f"GPD ajustado: ξ={self.xi:.4f}, σ={self.sigma:.4f}")
        
        return {
            'xi': self.xi,
            'sigma': self.sigma,
            'xi_se': self.std_errors[0] if self.std_errors is not None else None,
            'sigma_se': self.std_errors[1] if self.std_errors is not None else None,
            'threshold': self.threshold,
            'n_exceedances': self.n_exceedances,
            'exceedance_rate': self.n_exceedances / len(data),
            'method': method,
            'converged': self.converged,
            'upper_endpoint': self.upper_endpoint,
            **diagnostics
        }
    
    def _fit_mle(self):
        """MLE com tratamento de bounds"""
        xi0, sigma0 = self._pwm_estimates()
        
        def neg_loglik(params):
            xi, sigma = params
            if sigma <= 0:
                return 1e10
            
            x = self.exceedances
            n = len(x)
            
            if abs(xi) < 1e-10:
                ll = -n * np.log(sigma) - np.sum(x) / sigma
            else:
                # Para ξ < 0, verifica upper bound
                if xi < 0:
                    if np.any(x > -sigma/xi):
                        return 1e10
                
                z = 1 + xi * x / sigma
                if np.any(z <= 0):
                    return 1e10
                
                ll = -n * np.log(sigma) - (1/xi + 1) * np.sum(np.log(z))
            
            return -ll
        
        try:
            # Bounds mais flexíveis
            result = optimize.minimize(
                neg_loglik,
                x0=[xi0, sigma0],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5), (1e-6, None)]
            )
            
            if result.success:
                self.xi, self.sigma = result.x
                self.converged = True
            else:
                logger.warning(f"MLE não convergiu: {result.message}. Usando PWM")
                self.xi, self.sigma = xi0, sigma0
                self.converged = False
        except Exception as e:
            logger.error(f"Erro na MLE: {e}. Usando PWM")
            self.xi, self.sigma = xi0, sigma0
            self.converged = False
    
    def _fit_pwm(self):
        """Probability Weighted Moments"""
        self.xi, self.sigma = self._pwm_estimates()
        self.converged = True
    
    def _pwm_estimates(self) -> Tuple[float, float]:
        """PWM estimators"""
        x = np.sort(self.exceedances)
        n = len(x)
        
        a0 = np.mean(x)
        a1 = np.sum([(n - i) / (n - 1) * x[i] for i in range(n)]) / n
        
        xi_hat = 2 - a0 / (a0 - 2*a1)
        sigma_hat = 2 * a0 * a1 / (a0 - 2*a1)
        
        # Sanitize
        if np.isnan(xi_hat) or np.isinf(xi_hat):
            xi_hat = 0.1
        if np.isnan(sigma_hat) or sigma_hat <= 0:
            sigma_hat = np.std(x)
        
        # Bounds mais amplos 
        xi_hat = np.clip(xi_hat, -0.8, 0.8)
        
        return xi_hat, sigma_hat
    
    def _calculate_std_errors_analytical(self):
        """
        Hessiana analítica para GPD baseada em Smith (1985)
        """
        try:
            x = self.exceedances
            n = len(x)
            xi, sigma = self.xi, self.sigma
            
            if abs(xi) < 1e-10:
                # Caso exponencial
                H11 = n / sigma**2
                H12 = np.sum(x) / sigma**2
                H22 = n / sigma**2
                hessian = np.array([[H11, H12], [H12, H22]])
            else:
                z = 1 + xi * x / sigma
                
                # Elementos da Hessiana
                H11 = n / xi**2 - 2 * (1/xi + 1) * np.sum(1/z) + \
                      (1/xi + 1) * (1/xi + 2) * np.sum((x/sigma)**2 / z**2)
                
                H12 = -(1/xi + 1) * np.sum(x / (sigma * z)) + \
                      (1/xi + 1) * (1/xi + 2) * np.sum(x**2 / (sigma**2 * z**2))
                
                H22 = -n / sigma**2 + (1/xi + 1) * (1/xi + 2) * np.sum((xi * x / sigma**2)**2 / z**2)
                
                hessian = np.array([[H11, H12], [H12, H22]])
            
            # Covariância = inversa da Hessiana
            cov_matrix = np.linalg.inv(hessian)
            self.hessian = hessian
            
            std_errs = np.sqrt(np.abs(np.diag(cov_matrix)))
            
            if np.all(np.isfinite(std_errs)) and np.all(std_errs > 0):
                self.std_errors = std_errs
            else:
                logger.warning("Erros padrão analíticos inválidos, usando numéricos")
                self._calculate_std_errors_numerical()
        except Exception as e:
            logger.warning(f"Erro no cálculo analítico: {e},  usando numérico")
            self._calculate_std_errors_numerical()
    
    def _calculate_std_errors_numerical(self):
        """Hessiana numérica via diferenças finitas"""
        try:
            def neg_loglik(params):
                xi, sigma = params
                if sigma <= 0:
                    return 1e10
                
                x = self.exceedances
                n = len(x)
                
                if abs(xi) < 1e-10:
                    return n * np.log(sigma) + np.sum(x) / sigma
                
                z = 1 + xi * x / sigma
                if np.any(z <= 0):
                    return 1e10
                
                return n * np.log(sigma) + (1/xi + 1) * np.sum(np.log(z))
            
            # Hessiana numérica
            from scipy.optimize import approx_fprime
            
            params = np.array([self.xi, self.sigma])
            epsilon = np.sqrt(np.finfo(float).eps)
            
            # Calcula Hessiana por diferenças finitas de segunda ordem
            hessian = np.zeros((2, 2))
            
            def grad_func(p):
                return approx_fprime(p, neg_loglik, epsilon)
            
            for i in range(2):
                for j in range(2):
                    ei = np.zeros(2)
                    ej = np.zeros(2)
                    ei[i] = epsilon
                    ej[j] = epsilon
                    
                    hessian[i, j] = (
                        neg_loglik(params + ei + ej) -
                        neg_loglik(params + ei - ej) -
                        neg_loglik(params - ei + ej) +
                        neg_loglik(params - ei - ej)
                    ) / (4 * epsilon**2)
            
            cov_matrix = np.linalg.inv(hessian)
            self.hessian = hessian
            
            std_errs = np.sqrt(np.abs(np.diag(cov_matrix)))
            
            if np.all(np.isfinite(std_errs)) and np.all(std_errs > 0):
                self.std_errors = std_errs
            else:
                logger.warning("Erros padrão numéricos inválidos")
                self.std_errors = None
        except Exception as e:
            logger.warning(f"Não foi possível calcular erros padrão: {e}")
            self.std_errors = None
    
    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        seed: Optional[int] = None
    ) -> Dict:
        """Bootstrap paramétrico para intervalos de confiança"""
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        if seed is not None:
            np.random.seed(seed)
        
        xi_boot = []
        sigma_boot = []
        
        for _ in range(n_bootstrap):
            # Simula dados da GPD ajustada
            boot_data = self.simulate(self.n_exceedances)
            
            try:
                temp_gpd = GPD()
                temp_gpd.fit(
                    boot_data + self.threshold,
                    self.threshold,
                    method=self.method,
                    return_std_errors=False
                )
                
                if temp_gpd.converged:
                    xi_boot.append(temp_gpd.xi)
                    sigma_boot.append(temp_gpd.sigma)
            except:
                continue
        
        xi_boot = np.array(xi_boot)
        sigma_boot = np.array(sigma_boot)
        
        return {
            'xi_ci': (np.percentile(xi_boot, 100*alpha/2), 
                     np.percentile(xi_boot, 100*(1-alpha/2))),
            'sigma_ci': (np.percentile(sigma_boot, 100*alpha/2),
                        np.percentile(sigma_boot, 100*(1-alpha/2))),
            'xi_boot': xi_boot,
            'sigma_boot': sigma_boot
        }
    
    def _diagnostics(self) -> Dict:
        """Critérios de informação"""
        ll = self._loglikelihood(self.exceedances, self.xi, self.sigma)
        
        n = self.n_exceedances
        k = 2  # número de parâmetros
        
        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll
        
        return {
            'loglikelihood': ll,
            'aic': aic,
            'bic': bic
        }
    
    def _loglikelihood(self, x: np.ndarray, xi: float, sigma: float) -> float:
        """Log-verossimilhança"""
        n = len(x)
        
        if abs(xi) < 1e-10:
            return -n * np.log(sigma) - np.sum(x) / sigma
        else:
            z = 1 + xi * x / sigma
            if np.any(z <= 0):
                return -np.inf
            return -n * np.log(sigma) - (1/xi + 1) * np.sum(np.log(z))
    
    def goodness_of_fit(self, plot: bool = True) -> Dict:
        """
        Testes de qualidade do ajuste
        QQ-plot
        PP-plot  
        Kolmogorov-Smirnov
        Anderson-Darling
        """
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        x = self.exceedances
        n = len(x)
        
        # Valores teóricos e empíricos
        x_sorted = np.sort(x)
        empirical_cdf = np.arange(1, n+1) / (n+1)
        theoretical_cdf = self.cdf(x_sorted)
        
        # Quantis teóricos
        theoretical_quantiles = self.quantile(empirical_cdf)
        
        # Teste KS
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        ks_pvalue = stats.kstest(x, lambda y: self.cdf(y)).pvalue
        
        # Teste Anderson-Darling
        ad_stat = -n - np.sum((2*np.arange(1, n+1) - 1) / n * 
                              (np.log(theoretical_cdf) + 
                               np.log(1 - theoretical_cdf[::-1])))
        
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # QQ-plot
            ax1.scatter(theoretical_quantiles, x_sorted, alpha=0.6, s=30)
            lims = [min(theoretical_quantiles.min(), x_sorted.min()),
                    max(theoretical_quantiles.max(), x_sorted.max())]
            ax1.plot(lims, lims, 'r--', lw=2, label='45° line')
            ax1.set_xlabel('Theoretical Quantiles')
            ax1.set_ylabel('Empirical Quantiles')
            ax1.set_title('QQ-Plot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PP-plot
            ax2.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=30)
            ax2.plot([0, 1], [0, 1], 'r--', lw=2, label='45° line')
            ax2.set_xlabel('Theoretical CDF')
            ax2.set_ylabel('Empirical CDF')
            ax2.set_title('PP-Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'ad_statistic': ad_stat,
            'empirical_cdf': empirical_cdf,
            'theoretical_cdf': theoretical_cdf,
            'theoretical_quantiles': theoretical_quantiles
        }
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function"""
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        x = np.atleast_1d(x)
        
        # Para ξ < 0, verifica upper bound
        if self.xi < 0:
            x = np.where(x > -self.sigma/self.xi, np.nan, x)
        
        if abs(self.xi) < 1e-10:
            return (1/self.sigma) * np.exp(-x/self.sigma)
        else:
            z = 1 + self.xi * x / self.sigma
            z = np.maximum(z, 1e-10)
            return (1/self.sigma) * z ** (-(1/self.xi + 1))
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function"""
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        x = np.atleast_1d(x)
        
        # Para ξ < 0, verifica upper bound
        if self.xi < 0:
            x = np.where(x > -self.sigma/self.xi, 1.0, x)
        
        if abs(self.xi) < 1e-10:
            return 1 - np.exp(-x/self.sigma)
        else:
            z = 1 + self.xi * x / self.sigma
            z = np.maximum(z, 1e-10)
            return 1 - z ** (-1/self.xi)
    
    def quantile(self, p: float) -> float:
        """Quantile function"""
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        if abs(self.xi) < 1e-10:
            return -self.sigma * np.log(1-p)
        else:
            q = (self.sigma / self.xi) * ((1-p)**(-self.xi) - 1)
            
            # Para ξ < 0, verifica upper bound
            if self.xi < 0 and q > -self.sigma/self.xi:
                return -self.sigma/self.xi
            
            return q
    
    def var(self, p: float, n_obs: int) -> float:
        """
        Value at Risk, tratamento para ξ < 0
        """
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        Fu = self.n_exceedances / n_obs
        
        if p > Fu and not np.isclose(p, Fu, rtol=0.05):
            logger.warning(f"p={p:.4f} fora do range (Fu={Fu:.4f})")
            return self.threshold
        
        # VaR condicional
        if abs(self.xi) < 1e-10:
            q = self.threshold - self.sigma * np.log((n_obs / self.n_exceedances) * p)
        else:
            q = self.threshold + (self.sigma / self.xi) * (
                ((n_obs / self.n_exceedances) * p) ** (-self.xi) - 1
            )
        
        # Para ξ < 0, limita ao upper endpoint
        if self.xi < 0 and q > self.upper_endpoint:
            logger.warning(f"VaR truncado no upper endpoint: {self.upper_endpoint:.4f}")
            return self.upper_endpoint
        
        return q
    
    def expected_shortfall(self, p: float, n_obs: int) -> float:
        """
        Expected Shortfall (CVaR), tratamento para ξ < 0
        """
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        Fu = self.n_exceedances / n_obs
        
        if p > Fu and not np.isclose(p, Fu, rtol=0.05):
            logger.warning(f"p={p:.4f} fora do range (Fu={Fu:.4f})")
            return self.threshold
        
        var_p = self.var(p, n_obs)
        
        if self.xi >= 1:
            logger.warning("ξ >= 1: ES não definido, retornando VaR")
            return var_p
        
        # Fórmula para ES
        es = (var_p + self.sigma - self.xi * self.threshold) / (1 - self.xi)
        
        # Para ξ < 0, limita ao upper endpoint
        if self.xi < 0 and es > self.upper_endpoint:
            logger.warning(f"ES truncado no upper endpoint: {self.upper_endpoint:.4f}")
            return self.upper_endpoint
        
        return es
    

    def simulate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Simula dados da GPD"""
        if self.xi is None:
            raise ValueError("Ajuste o modelo primeiro")
        
        if seed is not None:
            np.random.seed(seed)
        
        u = np.random.uniform(0, 1, n_samples)
        
        if abs(self.xi) < 1e-10:
            samples = -self.sigma * np.log(1 - u)
        else:
            samples = (self.sigma / self.xi) * ((1 - u)**(-self.xi) - 1)
        
        # Para ξ < 0, trunca no upper endpoint
        if self.xi < 0:
            upper = -self.sigma / self.xi
            samples = np.minimum(samples, upper)
        
        return samples
    
    def return_level(
        self,
        return_period: float,
        n_obs: int,
        obs_per_period: int = 252
    ) -> float:
        """Return level para dado período de retorno"""
        n_u = self.n_exceedances / n_obs
        p = 1 / (return_period * obs_per_period)
        p_adjusted = p / n_u
        
        return self.var(p_adjusted, n_obs)

#funções auxiliares

def compare_threshold_methods(
    data: np.ndarray,
    methods: List[str] = ['mrl', 'stability', 'automated'],
    plot: bool = True
) -> pd.DataFrame:
    """
    Compara diferentes métodos de seleção de threshold
    """
    results = []
    
    for method in methods:
        gpd = GPD()
        try:
            threshold = gpd.select_threshold(data, method=method, plot=False)
            gpd.fit(data, threshold, method='mle')
            
            results.append({
                'method': method,
                'threshold': threshold,
                'n_exceedances': gpd.n_exceedances,
                'xi': gpd.xi,
                'sigma': gpd.sigma,
                'aic': gpd._diagnostics()['aic'],
                'bic': gpd._diagnostics()['bic']
            })
        except Exception as e:
            logger.warning(f"Método {method} falhou: {e}")
    
    df = pd.DataFrame(results)
    
    if plot and len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].bar(df['method'], df['threshold'])
        axes[0, 0].set_title('Threshold por Método')
        axes[0, 0].set_ylabel('Threshold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(df['method'], df['n_exceedances'])
        axes[0, 1].set_title('Número de Exceedances')
        axes[0, 1].set_ylabel('N Exceedances')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(df['method'], df['xi'])
        axes[1, 0].set_title('Shape Parameter (ξ)')
        axes[1, 0].set_ylabel('ξ')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(df['method'], df['aic'], alpha=0.7, label='AIC')
        axes[1, 1].bar(df['method'], df['bic'], alpha=0.7, label='BIC')
        axes[1, 1].set_title('Critérios de Informação')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return df


def diagnostic_plots(gpd: GPD, data: np.ndarray, full_analysis: bool = True):
    """
    Plots GPD
    """
    if gpd.xi is None:
        raise ValueError("GPD não ajustado")
    
    if full_analysis:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # histograma densidade 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gpd.exceedances, bins=30, density=True, alpha=0.6, 
             edgecolor='black', label='Dados')
    x_plot = np.linspace(0, gpd.exceedances.max(), 200)
    ax1.plot(x_plot, gpd.pdf(x_plot), 'r-', lw=2, label='GPD ajustada')
    ax1.set_xlabel('Exceedances')
    ax1.set_ylabel('Densidade')
    ax1.set_title('Densidade Empírica vs Teórica')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QQ-plot
    ax2 = fig.add_subplot(gs[0, 1])
    x_sorted = np.sort(gpd.exceedances)
    n = len(x_sorted)
    empirical_cdf = np.arange(1, n+1) / (n+1)
    theoretical_quantiles = gpd.quantile(empirical_cdf)
    
    ax2.scatter(theoretical_quantiles, x_sorted, alpha=0.6, s=30)
    lims = [min(theoretical_quantiles.min(), x_sorted.min()),
            max(theoretical_quantiles.max(), x_sorted.max())]
    ax2.plot(lims, lims, 'r--', lw=2)
    ax2.set_xlabel('Quantis Teóricos')
    ax2.set_ylabel('Quantis Empíricos')
    ax2.set_title('QQ-Plot')
    ax2.grid(True, alpha=0.3)
    
    # PP-plot
    ax3 = fig.add_subplot(gs[0, 2])
    theoretical_cdf = gpd.cdf(x_sorted)
    ax3.scatter(theoretical_cdf, empirical_cdf, alpha=0.6, s=30)
    ax3.plot([0, 1], [0, 1], 'r--', lw=2)
    ax3.set_xlabel('CDF Teórica')
    ax3.set_ylabel('CDF Empírica')
    ax3.set_title('PP-Plot')
    ax3.grid(True, alpha=0.3)
    
    # Nível de retorno
    ax4 = fig.add_subplot(gs[1, 0])
    return_periods = np.logspace(0, 3, 50)  # 1 a 1000 períodos
    return_levels = [gpd.return_level(rp, len(data)) for rp in return_periods]
    
    ax4.semilogx(return_periods, return_levels, 'b-', lw=2)
    ax4.set_xlabel('Período de Retorno')
    ax4.set_ylabel('Return Level')
    ax4.set_title('Nível de Retorno')
    ax4.grid(True, alpha=0.3)
    
    # Resíduos
    ax5 = fig.add_subplot(gs[1, 1])
    theoretical_cdf = gpd.cdf(gpd.exceedances)
    residuals = -np.log(1 - theoretical_cdf)  # Exponential transform
    theoretical_exp = np.sort(residuals)
    empirical_exp = -np.log(1 - np.arange(1, n+1)/(n+1))
    
    ax5.scatter(theoretical_exp, empirical_exp, alpha=0.6, s=30)
    lims = [0, max(theoretical_exp.max(), empirical_exp.max())]
    ax5.plot(lims, lims, 'r--', lw=2)
    ax5.set_xlabel('Resíduos Teóricos (Exp)')
    ax5.set_ylabel('Resíduos Empíricos (Exp)')
    ax5.set_title('QQ-Plot Exponencial (Resíduos)')
    ax5.grid(True, alpha=0.3)
    
    # Exceedances over time
    ax6 = fig.add_subplot(gs[1, 2])
    exceedance_indices = np.where(data > gpd.threshold)[0]
    ax6.scatter(exceedance_indices, data[exceedance_indices], 
                alpha=0.6, s=20, c='red', label='Exceedances')
    ax6.axhline(y=gpd.threshold, color='blue', linestyle='--', 
                lw=2, label=f'Threshold = {gpd.threshold:.3f}')
    ax6.set_xlabel('Index')
    ax6.set_ylabel('Valor')
    ax6.set_title('Exceedances ao Longo do Tempo')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    if full_analysis:
        # Parameter stability (varying threshold)
        ax7 = fig.add_subplot(gs[2, 0])
        n_thresh = 30
        sorted_data = np.sort(data)
        n_data = len(data)
        min_exc = 50
        
        thresholds = sorted_data[np.linspace(0, n_data - min_exc, n_thresh, dtype=int)]
        xi_vals = []
        
        for u in thresholds:
            try:
                temp_gpd = GPD()
                temp_gpd.fit(data, u, method='mle', return_std_errors=False)
                if temp_gpd.converged:
                    xi_vals.append(temp_gpd.xi)
                else:
                    xi_vals.append(np.nan)
            except:
                xi_vals.append(np.nan)
        
        ax7.plot(thresholds, xi_vals, 'b-', lw=2)
        ax7.axhline(y=gpd.xi, color='r', linestyle='--', 
                    lw=2, label=f'ξ estimado = {gpd.xi:.3f}')
        ax7.set_xlabel('Threshold')
        ax7.set_ylabel('ξ')
        ax7.set_title('Estabilidade')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # VaR e ES for diferentes níveis de confiança
        ax8 = fig.add_subplot(gs[2, 1])
        confidence_levels = np.linspace(0.90, 0.999, 50)
        p_levels = 1 - confidence_levels
        
        var_vals = [gpd.var(p, len(data)) for p in p_levels]
        es_vals = [gpd.expected_shortfall(p, len(data)) for p in p_levels]
        
        ax8.plot(confidence_levels, var_vals, 'b-', lw=2, label='VaR')
        ax8.plot(confidence_levels, es_vals, 'r-', lw=2, label='ES')
        ax8.set_xlabel('Nível de Confiança')
        ax8.set_ylabel('Valor')
        ax8.set_title('VaR e Expected Shortfall')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Confidence intervals 
        ax9 = fig.add_subplot(gs[2, 2])
        if gpd.std_errors is not None:
            labels = ['ξ', 'σ']
            estimates = [gpd.xi, gpd.sigma]
            errors = gpd.std_errors
            
            x_pos = np.arange(len(labels))
            ax9.bar(x_pos, estimates, yerr=1.96*errors, 
                    capsize=10, alpha=0.7, edgecolor='black')
            ax9.set_xticks(x_pos)
            ax9.set_xticklabels(labels)
            ax9.set_ylabel('Valor')
            ax9.set_title('Parâmetros com IC 95%')
            ax9.grid(True, alpha=0.3, axis='y')
        else:
            ax9.text(0.5, 0.5, 'Erros padrão\nnão disponíveis', 
                    ha='center', va='center', transform=ax9.transAxes,
                    fontsize=12)
            ax9.set_title('Intervalos de Confiança')
    
    plt.suptitle(f'Diagnósticos GPD (ξ={gpd.xi:.4f}, σ={gpd.sigma:.4f})', 
                 fontsize=14, y=0.995)
    plt.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    np.random.seed(42)
    
    # Simula dados com cauda pesada
    n = 5000
    data = stats.t.rvs(df=3, size=n) * 2 + 1
  
    print("Seleção Automática de Threshold")
    
    # Compara métodos de seleção de threshold
    comparison = compare_threshold_methods(data, plot=True)
    print("\nComparação de Métodos:")
    print(comparison.to_string(index=False))
    
    print("Ajuste GPD com Análise Completa")
    
    # Usa threshold automático
    gpd = GPD()
    threshold = gpd.select_threshold(data, method='automated', plot=False)
    
    # Ajusta modelo
    results = gpd.fit(data, threshold, method='mle', use_analytical_hessian=True)
    
    print(f"\nResultados do Ajuste:")
    print(f"Threshold: {results['threshold']:.4f}")
    print(f"N exceedances: {results['n_exceedances']} ({results['exceedance_rate']*100:.2f}%)")
    print(f"ξ = {results['xi']:.4f} ± {results['xi_se']:.4f}")
    print(f"σ = {results['sigma']:.4f} ± {results['sigma_se']:.4f}")
    print(f"AIC: {results['aic']:.2f}, BIC: {results['bic']:.2f}")
    print(f"Upper endpoint: {results['upper_endpoint']}")
    
    print("Testes de Qualidade do Ajuste")
  
    
    # Goodness of fit
    gof = gpd.goodness_of_fit(plot=True)
    print(f"\nTestes de Ajuste:")
    print(f"Kolmogorov-Smirnov: stat={gof['ks_statistic']:.4f}, p-value={gof['ks_pvalue']:.4f}")
    print(f"Anderson-Darling: stat={gof['ad_statistic']:.4f}")
    
    print("Intervalos de Confiança Bootstrap")
    
    # Bootstrap CI
    boot_results = gpd.bootstrap_ci(n_bootstrap=1000, seed=42)
    print(f"\nIntervalos de Confiança (Bootstrap 95%):")
    print(f"ξ: ({boot_results['xi_ci'][0]:.4f}, {boot_results['xi_ci'][1]:.4f})")
    print(f"σ: ({boot_results['sigma_ci'][0]:.4f}, {boot_results['sigma_ci'][1]:.4f})")
    

    print("Métricas de Risco")
    
    print(f"\nValue at Risk:")
    print(f"VaR 95%: {gpd.var(0.05, n):.4f}")
    print(f"VaR 99%: {gpd.var(0.01, n):.4f}")
    print(f"VaR 99.9%: {gpd.var(0.001, n):.4f}")
    
    print(f"\nExpected Shortfall:")
    print(f"ES 95%: {gpd.expected_shortfall(0.05, n):.4f}")
    print(f"ES 99%: {gpd.expected_shortfall(0.01, n):.4f}")
    print(f"ES 99.9%: {gpd.expected_shortfall(0.001, n):.4f}")
    
    print(f"\nReturn Levels:")
    print(f"10 anos: {gpd.return_level(10, n):.4f}")
    print(f"50 anos: {gpd.return_level(50, n):.4f}")
    print(f"100 anos: {gpd.return_level(100, n):.4f}")
    
    print("\n" + "="*70)
    print("Outros Plots")
    print("="*70)
    
    diagnostic_plots(gpd, data, full_analysis=True)
    
    print("Teste com ξ < 0 (Cauda Leve)")
    
    # Simula dados com cauda leve Beta transformada
    data_light = -np.log(stats.beta.rvs(2, 5, size=n))
    
    gpd_light = GPD()
    threshold_light = gpd_light.select_threshold(data_light, method='automated', plot=False)
    results_light = gpd_light.fit(data_light, threshold_light, method='mle')
    
    print(f"\nResultados (Cauda Leve):")
    print(f"ξ = {results_light['xi']:.4f} (< 0 indica cauda leve)")
    print(f"σ = {results_light['sigma']:.4f}")
    print(f"Upper endpoint: {results_light['upper_endpoint']:.4f}")
    print(f"VaR 99%: {gpd_light.var(0.01, n):.4f}")
    print(f"ES 99%: {gpd_light.expected_shortfall(0.01, n):.4f}")
    

    print("Aplicação em Retornos Financeiros")
    
    # Simula retornos financeiros
    returns = np.random.standard_t(4, size=n) * 0.02
    
    # Analisa cauda negativa 
    losses = -returns
    
    gpd_losses = GPD()
    threshold_losses = gpd_losses.select_threshold(losses, method='stability', plot=False)
    results_losses = gpd_losses.fit(losses, threshold_losses, method='mle')
    
    print(f"\nAnálise de Perdas (Cauda Negativa):")
    print(f"Threshold: {threshold_losses:.4f}")
    print(f"ξ = {results_losses['xi']:.4f}")
    print(f"σ = {results_losses['sigma']:.4f}")
    print(f"\nRisco para 1 dia:")
    print(f"VaR 95% (1 dia): {gpd_losses.var(0.05, n):.4f}")
    print(f"VaR 99% (1 dia): {gpd_losses.var(0.01, n):.4f}")
    print(f"ES 99% (1 dia): {gpd_losses.expected_shortfall(0.01, n):.4f}")
    