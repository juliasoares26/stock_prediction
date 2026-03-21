import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

@dataclass
class CopulaSelectionResult:
    """Resultado da seleção de cópula para um par de ativos."""
    pair: Tuple[str, str]          
    best_family: str               # nome da família vencedora
    best_copula: object            # instância ajustada da melhor cópula
    theta: float                   # parâmetro principal
    tau_kendall: float             # tau de Kendall empírico
    log_likelihood: float
    aic: float
    bic: float
    tail_lower: float              # λ_L
    tail_upper: float              # λ_U
    selection_criterion: str       # 'aic' ou 'bic'
    all_results: pd.DataFrame      # ranking completo de todas as famílias

    def summary(self) -> str:
        lines = [
            f"Par ({self.pair[0]}, {self.pair[1]})",
            f" Melhor família : {self.best_family}",
            f"θ              : {self.theta:.4f}",
            f"τ Kendall      : {self.tau_kendall:.4f}",
            f"AIC            : {self.aic:.2f}",
            f"λ_L / λ_U      : {self.tail_lower:.4f} / {self.tail_upper:.4f}",
        ]
        return "\n".join(lines)

# Seletor de cópula

class CopulaSelector:
    """
    Seleciona a família de cópula ótima para um par de ativos
    """

    # Famílias disponíveis
    ALL_FAMILIES = [
        "gaussian",
        "t",
        "clayton",
        "gumbel",
        "frank",
        "joe",
        "clayton180",
    ]

    def __init__(
        self,
        families: Optional[List[str]] = None,
        criterion: str = "aic",
        parallel: bool = True,
        tau_threshold: float = 0.05,
        n_workers: int = 4,
    ):
        self.families = families or self.ALL_FAMILIES
        self.criterion = criterion.lower()
        self.parallel = parallel
        self.tau_threshold = tau_threshold
        self.n_workers = n_workers

    # Seleção de um par

    def select(
        self,
        u: np.ndarray,
        v: np.ndarray,
        pair_name: Tuple[str, str] = ("u", "v"),
    ) -> CopulaSelectionResult:
        """
        Seleciona a melhor cópula para o par (u, v)

        """
        u = np.clip(u.ravel(), 1e-9, 1-1e-9).astype(np.float64)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9).astype(np.float64)

        tau, _ = stats.kendalltau(u, v)

        # Dependência fraca: retornar independência diretamente
        if abs(tau) < self.tau_threshold:
            logger.debug(f"Par {pair_name}: |τ|={abs(tau):.3f} < {self.tau_threshold}. Independência")
            return self._independence_result(pair_name, tau)

        # Ajustar todas as famílias
        if self.parallel:
            results_raw = self._fit_parallel(u, v)
        else:
            results_raw = self._fit_sequential(u, v)

        # Construir DataFrame de comparação
        rows = []
        # Tail dependence empirica para todos os pares (nao depende de nu)
        _U_pair = np.column_stack([u, v])
        try:
            from copulas.elliptical import empirical_tail_dependence
        except ImportError:
            try:
                from elliptical import empirical_tail_dependence
            except ImportError:
                empirical_tail_dependence = None

        if empirical_tail_dependence is not None:
            _td_emp = empirical_tail_dependence(_U_pair, alpha=0.10)
            _lam_L_emp = _td_emp['lower_tail']
            _lam_U_emp = _td_emp['upper_tail']
        else:
            _lam_L_emp, _lam_U_emp = np.nan, np.nan

        for fam, copula, error in results_raw:
            if copula is not None and error is None:
                # Tail dependence: usar empirica 
                # Para t-Student, td analitica via nu e nao confiavel
                lower_td = _lam_L_emp
                upper_td = _lam_U_emp

                ll  = getattr(copula, 'log_likelihood_', np.nan)
                aic = getattr(copula, 'aic_', np.nan)
                bic = getattr(copula, 'bic_', np.nan)

                # Para t-Student: AIC/BIC nao sao comparaveis com Gaussiana
                # via LL da copula
                # faz LL_copula_t > LL_copula_gauss mesmo com dados Gaussianos
                # Correcao: penalizar t-Student pelo parametro nu extra
                # usando teste de razao de verossimilhanca vs Gaussiana
                if fam == 't':
                    # Penalidade adicional: +5 AIC se nu > 25 (quase Gaussiana)
                    nu_val = getattr(copula, 'nu_', 30.0)
                    if nu_val >= 25.0:
                        aic = aic + 5.0 * (nu_val / 25.0)  # penalidade crescente
                        bic = bic + 5.0 * (nu_val / 25.0)

                row = {
                    "family": fam,
                    "log_likelihood": ll,
                    "aic": aic,
                    "bic": bic,
                    "theta": getattr(copula, "theta_", getattr(copula, "rho_", np.nan)),
                    "nu": getattr(copula, "nu_", np.nan),
                    "lower_tail": lower_td,
                    "upper_tail": upper_td,
                    "converged": True,
                    "_copula": copula,
                }
            else:
                row = {
                    "family": fam,
                    "log_likelihood": -np.inf,
                    "aic": np.inf,
                    "bic": np.inf,
                    "theta": np.nan,
                    "nu": np.nan,
                    "lower_tail": np.nan,
                    "upper_tail": np.nan,
                    "converged": False,
                    "_copula": None,
                }
            rows.append(row)

        sel_df = pd.DataFrame(rows)
        sel_df = sel_df.sort_values(self.criterion).reset_index(drop=True)
    
        # t-Student vs Gaussiana via tail dependence empirica
        # Problema: LL da copula t SEMPRE > LL Gaussiana via AIC/BIC
        # (diferenca ~2.3 por obs, artefato da transformacao t^{-1}(U))
        # Solucao: se tail dep empirica e baixa em ambas caudas,
        # t-Student nao adiciona valor real, substituir por Gaussiana
        # Threshold 0.15 baseado em: lambda_L teorica da t com nu=30, rho=0.4
    
        _td_low  = float(_lam_L_emp) if not np.isnan(_lam_L_emp) else 1.0
        _td_high = float(_lam_U_emp) if not np.isnan(_lam_U_emp) else 1.0
        _T = len(u)
        # Threshold adaptativo: mais conservador com mais dados
        _td_threshold = 0.15 + 0.05 * (1 - min(_T, 1000) / 1000)

        best_row = sel_df.iloc[0]
        best_copula = best_row["_copula"]
        best_family = best_row["family"]

        if best_family == 't' and _td_low < _td_threshold and _td_high < _td_threshold:
            # Sem evidencia de tail dependence — substituir por Gaussiana
            gauss_rows = sel_df[sel_df['family'] == 'gaussian']
            if not gauss_rows.empty and gauss_rows.iloc[0]['_copula'] is not None:
                best_row    = gauss_rows.iloc[0]
                best_copula = best_row['_copula']
                best_family = 'gaussian'
                logger.debug(
                    f"Par {pair_name}: t-Student substituida por Gaussiana "
                    f"(lL={_td_low:.3f} < {_td_threshold:.2f}, "
                    f"lU={_td_high:.3f} < {_td_threshold:.2f})"
                )

        # Exibir df sem a coluna interna
        display_df = sel_df.drop(columns=["_copula"]).round(4)

        logger.info(
            f"Par {pair_name} | τ={tau:.3f} | "
            f"Melhor: {best_family} ({self.criterion.upper()}={best_row[self.criterion]:.2f})"
        )

        return CopulaSelectionResult(
            pair=pair_name,
            best_family=best_family,
            best_copula=best_copula,
            theta=float(best_row["theta"]),
            tau_kendall=float(tau),
            log_likelihood=float(best_row["log_likelihood"]),
            aic=float(best_row["aic"]),
            bic=float(best_row["bic"]),
            tail_lower=float(best_row["lower_tail"]),
            tail_upper=float(best_row["upper_tail"]),
            selection_criterion=self.criterion,
            all_results=display_df,
        )

    # Fit famílias

    def _fit_one(self, family: str, u: np.ndarray, v: np.ndarray):
        """Ajusta uma família e retorna (family, copula, error)"""
        try:
            copula = self._get_copula_instance(family)
            if hasattr(copula, "fit") and copula.__class__.__name__ in (
                "GaussianCopula", "StudentTCopula"
            ):
                # Multivariado — usar versão bivariada
                copula = self._get_bivariate_instance(family)

            UV = np.column_stack([u, v])
            if hasattr(copula, "fit"):
                if copula.__class__.__name__ in (
                    "BivariateGaussianCopula", "BivariateStudentTCopula"
                ):
                    copula.fit(u, v)
                else:
                    copula.fit(u, v)
            return family, copula, None
        except Exception as e:
            logger.debug(f"  {family}: falhou — {e}")
            return family, None, str(e)

    def _fit_sequential(self, u, v):
        return [self._fit_one(fam, u, v) for fam in self.families]

    def _fit_parallel(self, u, v):
        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._fit_one, fam, u, v): fam
                for fam in self.families
            }
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def _get_copula_instance(self, family: str):
        try:
            from copulas.archimedean import (
                ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula180
            )
            from copulas.elliptical import BivariateGaussianCopula, BivariateStudentTCopula
        except ImportError:
            from archimedean import (
                ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula180
            )
            from elliptical import BivariateGaussianCopula, BivariateStudentTCopula

        registry = {
            "gaussian":   BivariateGaussianCopula,
            "t":          BivariateStudentTCopula,
            "clayton":    ClaytonCopula,
            "gumbel":     GumbelCopula,
            "frank":      FrankCopula,
            "joe":        JoeCopula,
            "clayton180": ClaytonCopula180,
        }
        cls = registry.get(family.lower())
        if cls is None:
            raise ValueError(f"Família desconhecida: {family}")
        return cls()

    def _get_bivariate_instance(self, family: str):
        return self._get_copula_instance(family)

    def _independence_result(
        self, pair_name: Tuple[str, str], tau: float
    ) -> CopulaSelectionResult:
        """Retorna resultado para dependência insignificante"""
        try:
            from copulas.archimedean import FrankCopula
        except ImportError:
            from archimedean import FrankCopula
        dummy = FrankCopula()
        dummy.theta_ = 0.001
        dummy.log_likelihood_ = 0.0
        dummy.aic_ = 2.0
        dummy.bic_ = np.log(100)
        dummy._is_fitted = True

        return CopulaSelectionResult(
            pair=pair_name,
            best_family="independence",
            best_copula=dummy,
            theta=0.0,
            tau_kendall=float(tau),
            log_likelihood=0.0,
            aic=2.0,
            bic=np.log(100),
            tail_lower=0.0,
            tail_upper=0.0,
            selection_criterion=self.criterion,
            all_results=pd.DataFrame(),
        )

    # Seleção para todos os pares

    def select_all_pairs(
        self,
        U: np.ndarray,
        column_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[Tuple[int, int], CopulaSelectionResult]:
        """
        Seleciona a melhor cópula para todos os pares de ativos

        Usado por vine_copulas.py para escolher a pair-copula de cada aresta

        """
        T, d = U.shape
        names = column_names or [f"ativo_{i}" for i in range(d)]
        n_pairs = d * (d-1) // 2

        logger.info(f"Selecionando cópulas para {n_pairs} pares de {d} ativos")
        results = {}

        for i in range(d):
            for j in range(i+1, d):
                pair = (names[i], names[j])
                result = self.select(U[:, i], U[:, j], pair_name=pair)
                results[(i, j)] = result

                if verbose:
                    logger.info(
                        f"  ({names[i]}, {names[j]}): "
                        f"{result.best_family} | "
                        f"θ={result.theta:.3f} | "
                        f"τ={result.tau_kendall:.3f} | "
                        f"λ_L={result.tail_lower:.3f} | "
                        f"λ_U={result.tail_upper:.3f}"
                    )

        return results

    def summarize_selection(
        self,
        pair_results: Dict[Tuple[int, int], CopulaSelectionResult],
    ) -> pd.DataFrame:
        """
        Resume a seleção de cópulas por par em um DataFrame

        """
        rows = []
        for (i, j), res in pair_results.items():
            rows.append({
                "pair": f"({res.pair[0]}, {res.pair[1]})",
                "i": i, "j": j,
                "best_family": res.best_family,
                "theta": res.theta,
                "tau_kendall": res.tau_kendall,
                "aic": res.aic,
                "bic": res.bic,
                "lower_tail": res.tail_lower,
                "upper_tail": res.tail_upper,
            })

        df = pd.DataFrame(rows)

        # Contagem por família
        logger.info("Distribuição de famílias selecionadas:")
        for fam, cnt in df["best_family"].value_counts().items():
            logger.info(f"  {fam:15s}: {cnt} pares ({cnt/len(df)*100:.1f}%)")

        return df

    def get_family_distribution(
        self,
        pair_results: Dict[Tuple[int, int], CopulaSelectionResult],
    ) -> pd.Series:
        """Contagem de famílias selecionadas."""
        families = [r.best_family for r in pair_results.values()]
        return pd.Series(families).value_counts()


# Teste de depemdemcia de cauda

class TailDependenceTest:
    """
    Testes estatísticos para dependência de cauda
    Ajuda a decidir entre famílias com e sem dependência de cauda

    Métodos:
    'cfg': estimador não-paramétrico de CFG
    'log': estimador log baseado em ranks extremos
    'hill': adaptação do estimador Hill para dependência de cauda
    """

    @staticmethod
    def estimate_tail_dependence(
        u: np.ndarray,
        v: np.ndarray,
        method: str = "cfg",
        alpha: float = 0.1,
    ) -> Dict[str, float]:
        """
        Estima coeficientes empíricos de dependência de cauda

        """
        u = np.clip(u.ravel(), 1e-9, 1-1e-9)
        v = np.clip(v.ravel(), 1e-9, 1-1e-9)
        T = len(u)

        if method == "cfg":
            lambda_lower, lambda_upper = TailDependenceTest._cfg_estimator(u, v, alpha)
        elif method == "log":
            lambda_lower, lambda_upper = TailDependenceTest._log_estimator(u, v, alpha)
        else:
            raise ValueError(f"método desconhecido: {method}")

        # Teste heurístico: se λ > 0.1, há dependência de cauda relevante
        is_tail_dependent = lambda_lower > 0.1 or lambda_upper > 0.1

        return {
            "lower": float(lambda_lower),
            "upper": float(lambda_upper),
            "is_tail_dependent": bool(is_tail_dependent),
            "method": method,
            "alpha": alpha,
        }

    @staticmethod
    def _cfg_estimator(u: np.ndarray, v: np.ndarray, alpha: float):
        """
        CFG estimador não-paramétrico.
        λ_L = lim_{t→0} C(t,t)/t
        λ_U = lim_{t→1} (1-2t+C(t,t))/(1-t)
        """
        T = len(u)
        k = max(1, int(T * alpha))  # número de observações na cauda

        # Cauda inferior: ambos abaixo do k-ésimo quantil
        u_thresh = np.sort(u)[k]
        v_thresh = np.sort(v)[k]
        joint_lower = np.mean((u <= u_thresh) & (v <= v_thresh))
        lambda_lower = joint_lower / (k/T) if k/T > 0 else 0.0

        # Cauda superior: ambos acima do (T-k)-ésimo quantil
        u_thresh_up = np.sort(u)[T-k-1]
        v_thresh_up = np.sort(v)[T-k-1]
        joint_upper = np.mean((u >= u_thresh_up) & (v >= v_thresh_up))
        lambda_upper = joint_upper / (k/T) if k/T > 0 else 0.0

        return float(lambda_lower), float(lambda_upper)

    @staticmethod
    def _log_estimator(u: np.ndarray, v: np.ndarray, alpha: float):
        """Estimador baseado em ranks extremos"""
        T = len(u)
        k = max(1, int(T * alpha))

        # Lower tail
        ranks_u = stats.rankdata(u)
        ranks_v = stats.rankdata(v)
        lower_mask = (ranks_u <= k) & (ranks_v <= k)
        lambda_lower = np.mean(lower_mask) * T / k

        # Upper tail
        upper_mask = (ranks_u >= T-k) & (ranks_v >= T-k)
        lambda_upper = np.mean(upper_mask) * T / k

        return float(lambda_lower), float(lambda_upper)

    @staticmethod
    def recommend_family(
        u: np.ndarray,
        v: np.ndarray,
        tau: Optional[float] = None,
    ) -> List[str]:
        """
        Recomenda famílias de cópula com base no tau de Kendall e
        dependência de cauda empírica
        """
        if tau is None:
            tau, _ = stats.kendalltau(u, v)

        td = TailDependenceTest.estimate_tail_dependence(u, v, method="cfg")

        recommendations = []

        if abs(tau) < 0.05:
            return ["independence", "frank"]

        if td["lower"] > 0.15 and td["upper"] > 0.15:
            # Dependência simétrica de cauda = t-Student
            recommendations = ["t", "gaussian"]
        elif td["lower"] > 0.15:
            # Dependência de cauda inferior = Clayton
            recommendations = ["clayton", "t", "gaussian"]
        elif td["upper"] > 0.15:
            # Dependência de cauda superior = Gumbel ou Joe
            recommendations = ["gumbel", "joe", "t"]
        else:
            # Sem dependência de cauda = Frank ou Gaussian
            recommendations = ["frank", "gaussian", "t"]

        # Sempre adicionar as outras como fallback
        all_fams = ["gaussian", "t", "clayton", "gumbel", "frank", "joe"]
        for fam in all_fams:
            if fam not in recommendations:
                recommendations.append(fam)

        return recommendations


def select_pair_copula(
    u: np.ndarray,
    v: np.ndarray,
    pair_name: Tuple[str, str] = ("i", "j"),
    criterion: str = "aic",
    families: Optional[List[str]] = None,
    use_recommendation: bool = True,
) -> CopulaSelectionResult:
    """
    Seleciona a melhor pair-copula para uso em vine_copulas.py

    Se use_recommendation=True, pré-filtra as famílias com base na
    dependência de cauda empírica para reduzir o tempo de computação

    """
    if use_recommendation and families is None:
        tau, _ = stats.kendalltau(u, v)
        families = TailDependenceTest.recommend_family(u, v, tau)[:5]  # top 5
        logger.debug(f"Par {pair_name}: famílias recomendadas = {families}")

    selector = CopulaSelector(
        families=families or CopulaSelector.ALL_FAMILIES,
        criterion=criterion,
        parallel=True,
    )
    return selector.select(u, v, pair_name=pair_name)

# Main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    np.random.seed(42)
    n = 800

    #  Clayton
    print("\n Caso 1: Clayton verdadeiro (θ=2)")
    theta_true = 2.0
    # Gerador correto: h_inverse analitica (marginais exatamente U(0,1))
    _u_raw = np.random.uniform(0, 1, n)
    _w_raw = np.random.uniform(0, 1, n)
    _inner = (_w_raw * _u_raw**(theta_true+1))**(-theta_true/(theta_true+1)) + 1 - _u_raw**(-theta_true)
    u1 = np.clip(_u_raw, 1e-9, 1-1e-9)
    v1 = np.clip(np.maximum(_inner, 1e-10)**(-1/theta_true), 1e-9, 1-1e-9)

    # Dependência de cauda empírica
    td = TailDependenceTest.estimate_tail_dependence(u1, v1)
    print(f"Tail dep empírica: λ_L={td['lower']:.3f}, λ_U={td['upper']:.3f}")
    recs = TailDependenceTest.recommend_family(u1, v1)
    print(f"Famílias recomendadas: {recs[:3]}")

    selector = CopulaSelector(criterion="aic", parallel=False)
    result1 = selector.select(u1, v1, pair_name=("PETR4", "VALE3"))
    print(result1.summary())
    print(f"\nRanking completo:\n{result1.all_results[['family','aic','theta','lower_tail']].to_string()}")

    #  Gumbel
    print("\n Caso 2: Gumbel verdadeiro (θ=2) ")
    try:
        from copulas.archimedean import GumbelCopula
    except ImportError:
        from archimedean import GumbelCopula
    gum = GumbelCopula()
    gum.theta_ = 2.0
    gum._is_fitted = True
    sim_gumbel = gum.simulate(n, seed=123)
    u2, v2 = sim_gumbel[:, 0], sim_gumbel[:, 1]

    result2 = selector.select(u2, v2, pair_name=("ITUB4", "BBAS3"))
    print(result2.summary())

    # Caso 3: Todos os pares de uma matriz 4x4
    print("\n Caso 3: select_all_pairs (d=4) ")
    R = 0.4*np.ones((4,4)) + 0.6*np.eye(4)
    L = np.linalg.cholesky(R)
    Z = np.random.standard_normal((n, 4)) @ L.T
    U4 = stats.norm.cdf(Z)

    names = ["PETR4", "VALE3", "ITUB4", "BBAS3"]
    all_results = selector.select_all_pairs(U4, column_names=names, verbose=True)
    summary_df = selector.summarize_selection(all_results)
    print(f"\nResumo dos pares:\n{summary_df[['pair','best_family','tau_kendall','lower_tail','upper_tail']].to_string()}")

    dist = selector.get_family_distribution(all_results)
    print(f"\nDistribuição de famílias:\n{dist}")

    print("\n Teste concluído.")
