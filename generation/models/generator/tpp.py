import warnings

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

try:
    from hmmlearn.hmm import CategoricalHMM
    from tick.base import TimeFunction
    from tick.hawkes import HawkesEM, HawkesKernelTimeFunc, SimuHawkes
except ImportError:
    print("Could not import TPP libs")

from ...data.data_types import DataConfig, GenBatch, PredBatch
from . import BaseGenerator, ModelConfig


class BaselineHP(BaseGenerator):
    def __init__(self, data_conf: DataConfig, model_config: ModelConfig):
        super().__init__()
        self._feature_name = model_config.tpp.feature_name

    def forward(self, x: GenBatch):
        raise "No need to train BaselineHP."

    @staticmethod
    def marks_times_to_tick_stream(
        marks: np.ndarray, times: np.ndarray, keep_original_labels: bool = False
    ):
        """
        (marks, times) -> events для tick.

        Parameters
        ----------
        marks : 1-D int array
            Метки (event type) той же длины, что times.
        times : 1-D float array
            Timestamps
        keep_original_labels : bool, default False
            • False  → маппим уникальные метки к 0..K-1.
            • True   → сохраняем оригинальные номера (если они уже плотные и начинаются с 0).

        Returns
        -------
        events : list[list[np.ndarray]]
            events[realization_idx][node_idx] = np.array([...])
        label_map : dict
            Словарь {original_label: new_label}.  If keep_original_labels=False.
        """

        marks = np.asarray(marks)
        times = np.asarray(times, dtype=float)
        assert np.all(np.diff(times) >= 0), "Times must be strictly increasing"

        if keep_original_labels:
            uniq = np.unique(marks)
            if (uniq[0] != 0) or (uniq[-1] != len(uniq) - 1):
                raise ValueError(
                    "Marks should be a dense range from 0 to K-1, or use keep_original_labels=False"
                )
            new_marks = marks
            label_map = {k: k for k in uniq}
        else:
            uniq = np.unique(marks)
            label_map = {orig: i for i, orig in enumerate(uniq)}
            new_marks = np.vectorize(label_map.get)(marks)

        K = len(uniq)

        timestamps = [times[new_marks == k] for k in range(K)]

        return [timestamps], label_map

    @staticmethod
    def tick_stream_to_marks_times(
        future_ts: list[np.ndarray],
        label_map: dict[int, int] | None = None,
        restore_original_labels: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Обратное преобразование:
        [timestamps_node0, ..., timestamps_node_{K-1}]  →  (marks, times)

        Parameters
        ----------
        future_ts : list[np.ndarray]
            Содержимое events[0] из tick: таймстемпы по каждому узлу.
        label_map : dict {orig_label -> new_idx}, опционально
            Словарь, который вернулся из `marks_times_to_tick_stream`.
            Если нужно восстановить исходные метки.
        restore_original_labels : bool, default False
            • False → вернём «плотные» индексы 0..K-1.
            • True  → Inverse transform based on label_map - return original labeld.

        Returns
        -------
        marks : np.ndarray (int)
        times : np.ndarray (float)
        """
        times = np.concatenate(future_ts)
        marks = np.concatenate(
            [np.full(len(ts), idx, dtype=int) for idx, ts in enumerate(future_ts)]
        )

        order = np.argsort(times)
        times = times[order]
        marks = marks[order]

        if restore_original_labels:
            assert label_map is not None
            inv = {v: k for k, v in label_map.items()}
            marks = np.vectorize(inv.get)(marks)

        return marks, times

    @staticmethod
    def select_hawkes_params(
        events: list[list[np.ndarray]],
        supports: list[float],
        sizes: list[int],
        max_iter: int = 50,
        tol: float = 1e-5,
        n_threads: int = 4,
    ) -> tuple[float, int]:
        """
        Grid-search по (kernel_support, kernel_size),
        возвращает пару (best_support, best_size), дающую максимальное log‐likelihood.
        """
        best_ll = -float("inf")
        best_s, best_k = supports[0], sizes[0]

        for s in supports:
            for k in sizes:
                learner = HawkesEM(
                    kernel_support=s,
                    kernel_size=k,
                    max_iter=max_iter,
                    tol=tol,
                    n_threads=n_threads,
                )
                _ = learner.fit(events)  # EM возвращает итоговое log‐likelihood

                ll = learner.score(events)

                if ll > best_ll:
                    best_ll, best_s, best_k = ll, s, k

        return best_s, best_k

    @staticmethod
    def hawkes_simulate(
        self,
        events,
        gen_len,
        supports=(0.5, 1.0, 2.0),
        sizes=(20, 50, 100),
    ):

        best_s, best_k = self.select_hawkes_params(
            events, supports=list(supports), sizes=list(sizes)
        )

        learner = HawkesEM(
            kernel_support=best_s,
            kernel_size=best_k,
            max_iter=50,
            tol=1e-5,
            n_threads=4,
        )

        _ = learner.fit(events)

        d = learner.n_nodes
        t_grid = np.linspace(0, best_s, best_k + 1)
        dt = best_s / best_k
        kernels = [[None] * d for _ in range(d)]

        for i in range(d):
            for j in range(d):
                y_vals = np.append(learner.kernel[i, j], learner.kernel[i, j][-1])
                tf = TimeFunction(
                    (t_grid, y_vals), inter_mode=TimeFunction.InterConstRight, dt=dt
                )
                kernels[i][j] = HawkesKernelTimeFunc(time_function=tf)

        last_t = max(max(h) for h in events[0])

        simu = SimuHawkes(
            baseline=learner.baseline,
            kernels=kernels,
            max_jumps=gen_len,
            seed=42,
            verbose=False,
        )

        simu.simulate()
        future_ts = [ts + last_t for ts in simu.timestamps]

        return future_ts

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        timestamps = hist.time
        feature = hist[self._feature_name]
        lengths = hist.lengths
        L, B = timestamps.shape

        gen_marks = []
        gen_times = []
        for _, seq in enumerate(zip(feature.T, timestamps.T, lengths)):
            marks, times, seq_len = seq
            marks = marks[:seq_len].cpu().numpy()
            times = times[:seq_len].cpu().numpy()

            events, lbl = self.marks_times_to_tick_stream(
                marks, times, keep_original_labels=False
            )
            sample = self.hawkes_simulate(
                self,
                events,
                gen_len,
                supports=(1, 2, 3),  #
                sizes=(20, 30, 50),
            )
            smp_marks, smp_times = self.tick_stream_to_marks_times(
                sample, label_map=lbl, restore_original_labels=True
            )

            gen_marks.append(smp_marks)
            gen_times.append(smp_times)
        generated = PredBatch(
            lengths=np.ones(B) * gen_len,
            time=torch.tensor(np.stack(gen_times).T, dtype=torch.float32),
            num_features=None,
            num_features_names=None,
            cat_features={
                self._feature_name: torch.tensor(
                    np.stack(gen_marks).T, dtype=torch.float32
                )
            },
        )
        return generated.to_batch()


def free_params(K: int, M: int) -> int:
    """Число свободных параметров у CategoricalHMM с K состояний и M категорий."""
    return (K - 1) + K * (K - 1) + K * (M - 1)


def is_valid_hmm(model):
    return (model.transmat_.sum(axis=1) > 1e-12).all()


def select_hmm_by_bic(
    obs: np.ndarray,
    M: int,
    max_K: int = 10,
    n_iter: int = 100,
    tol: float = 1e-4,
    restarts: int = 1,
) -> CategoricalHMM:
    N = obs.shape[0]
    best_bic = float("inf")
    best_m = None

    for K in range(1, min(M, max_K) + 1):
        k = free_params(K, M)
        if k >= N:
            continue

        best_logL = -float("inf")
        best_candidate = None

        warnings.filterwarnings(
            "ignore",
            message="Some rows of transmat_ have zero sum",
            category=RuntimeWarning,
        )

        def safe_fit(model, X):
            """Обучает HMM и чинит нулевые строки."""
            model.fit(X)  # может наплодить нули
            rowsum = model.transmat_.sum(1)
            if (rowsum == 0).any():  # встретили мёртвый state
                dead = rowsum == 0
                K = model.n_components
                model.transmat_[dead] = 1.0 / K  # равномерное распределение вместо нуля
            return model

        for _ in range(restarts):
            model = CategoricalHMM(
                n_components=K,
                n_iter=n_iter,
                tol=tol,
                init_params="ste",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = safe_fit(model, obs)

            try:
                logL = model.score(obs)
            except ValueError:
                continue

            if logL > best_logL:
                best_logL = logL
                best_candidate = model

        bic = k * np.log(N) - 2 * best_logL
        if bic < best_bic:
            best_bic = bic
            best_m = best_candidate

    if best_m is None:
        print("no best_m, so n_components = 1 :(")
        best_m = CategoricalHMM(n_components=1, n_iter=n_iter, tol=tol)
        best_m.fit(obs)

    return best_m


class BaselineHMM(BaseGenerator):
    def __init__(self, data_conf: DataConfig, model_config: ModelConfig):
        super().__init__()
        self._feature_name = model_config.tpp.feature_name

    def forward(self, x: GenBatch):
        raise "No need to train BaselineHMM."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        feature = hist[self._feature_name]
        lengths = hist.lengths
        B = feature.shape[-1]

        gen_marks = []
        for _, seq in enumerate(zip(feature.T, lengths)):
            marks, seq_len = seq
            marks = marks[:seq_len].cpu().numpy()

            label_enc = LabelEncoder()
            cat_seq = label_enc.fit_transform(marks)
            obs = cat_seq.reshape(-1, 1)
            M = len(label_enc.classes_)

            best_model = select_hmm_by_bic(
                obs=obs, M=M, max_K=10, n_iter=50, tol=1e-4, restarts=3
            )

            X_new, _ = best_model.sample(gen_len)
            gen_marks.append(label_enc.inverse_transform(X_new.ravel()))

        generated = PredBatch(
            lengths=np.ones(B) * gen_len,
            time=hist.target_time,
            num_features=None,
            num_features_names=None,
            cat_features={
                self._feature_name: torch.tensor(
                    np.stack(gen_marks).T, dtype=torch.float32
                )
            },
        )
        return generated.to_batch()
