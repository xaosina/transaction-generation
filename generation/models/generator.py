from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from ebes.model import BaseModel
from ebes.model.seq2seq import Projection

from generation.models.autoencoders.vae import Decoder as VAE_Decoder
from generation.models.autoencoders.vae import Encoder as VAE_Encoder
from generation.models.autoencoders.vae import VaeConfig

from ..data.data_types import DataConfig, GenBatch, PredBatch, gather
from .encoders import AutoregressiveEncoder, EncoderConfig
from .preprocessor import PreprocessorConfig, create_preprocessor
from .reconstructors import ReconstructorBase
from sklearn.preprocessing import LabelEncoder
from hmmlearn.hmm import CategoricalHMM
from tick.hawkes import HawkesEM, HawkesKernelTimeFunc, SimuHawkes
from tick.base import TimeFunction


@dataclass(frozen=True)
class TPPConfig:
    feature_name: str = ""


@dataclass(frozen=True)
class ModelConfig:
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    vae: VaeConfig = field(default_factory=VaeConfig)
    tpp: TPPConfig = field(default_factory=TPPConfig)


class BaseGenerator(BaseModel):
    def forward(self, x: GenBatch) -> PredBatch: ...

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch: ...


class GroundTruthGenerator(BaseGenerator):
    """To check that all preprocessing is fine. Get perfect baseline."""

    def forward(self, x: GenBatch):
        raise "No need to train a GroundTruthGenerator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        assert hist.target_time.shape[0] == gen_len
        gen_batch = deepcopy(hist)
        gen_batch.append(gen_batch.get_target_batch())

        gen_batch.target_time = None
        gen_batch.target_num_features = None
        gen_batch.target_cat_features = None

        if with_hist:
            return gen_batch  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return gen_batch.tail(gen_len)


class BaselineHP(BaseGenerator):
    def __init__(self, model_config: ModelConfig):
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
            Timestamp'ы, НЕобязательно отсортированы.
        keep_original_labels : bool, default False
            • False  → маппим уникальные метки к 0..K-1 (tick так комфортнее).
            • True   → сохраняем оригинальные номера (если они уже плотные и начинаются с 0).

        Returns
        -------
        events : list[list[np.ndarray]]
            Формат tick: events[realization_idx][node_idx] = np.array([...])
            Здесь одна реализация, поэтому len(events) == 1.
        label_map : dict
            Словарь {оригинальная_метка: новая_метка}.  Нужен, если keep_original_labels=False.
        """

        marks = np.asarray(marks)
        times = np.asarray(times, dtype=float)

        if np.any(np.diff(times) < 0):
            breakpoint()
            raise ValueError("Times must be strictly increasing")

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
            Тот самый словарь, который вернул `marks_times_to_tick_stream`.
            Нужен, если хотите восстановить исходные (разреженные) метки.
        restore_original_labels : bool, default False
            • False → вернём «плотные» индексы 0..K-1.
            • True  → применим инвертированную label_map, чтобы вернуть
                    оригинальные номера меток.

        Returns
        -------
        marks : np.ndarray (int)
        times : np.ndarray (float)
            Уже отсортированы по времени (строго возрастающие).
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
    def hawkes_simulate(
        events,
    ):

        learner = HawkesEM(
            kernel_support=1.0, kernel_size=30, max_iter=50, tol=1e-5, n_threads=4
        )

        _ = learner.fit(events)

        d = learner.n_nodes
        support = learner.kernel_support  # 2.0
        k_size = learner.kernel_size  # 30
        t_grid = np.linspace(0, support, k_size + 1)  # k_size+1 точек
        dt = support / k_size
        kernels_tf = [[None] * d for _ in range(d)]

        for i in range(d):
            for j in range(d):
                y_vals = np.append(learner.kernel[i, j], learner.kernel[i, j][-1])
                tf = TimeFunction(
                    (t_grid, y_vals), inter_mode=TimeFunction.InterConstRight, dt=dt
                )
                kernels_tf[i][j] = HawkesKernelTimeFunc(time_function=tf)

        last_t = max(max(h) for h in events[0])

        simu = SimuHawkes(
            baseline=learner.baseline, kernels=kernels_tf, max_jumps=300, seed=42
        )
        simu.simulate()
        future_ts = [ts + last_t for ts in simu.timestamps]

        return future_ts

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        timestamps = hist.time
        feature = hist[self._feature_name]
        lengths = hist.lengths

        B = timestamps.shape[-1]

        gen_marks = []
        gen_times = []
        for _, seq in enumerate(zip(feature.T, timestamps.T, lengths)):
            marks, times, seq_len = seq
            marks = marks[:seq_len].numpy()
            times = times[:seq_len].numpy()

            events, lbl = self.marks_times_to_tick_stream(
                marks, times, keep_original_labels=False
            )

            sample = self.hawkes_simulate(events)
            smp_marks, smp_times = self.tick_stream_to_marks_times(
                sample, label_map=lbl, restore_original_labels=True
            )

            gen_marks.append(smp_marks)
            gen_times.append(smp_times)

        generated = PredBatch(
            lengths=np.ones(B) * gen_len,
            time=np.stack(gen_times),
            num_features=None,
            num_features_names=None,
            cat_features={self._feature_name: np.stack(gen_marks)},
        )
        return generated


class BaselineHMM(BaseGenerator):
    def __init__(self, model_config: ModelConfig):
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
            marks = marks[:seq_len].numpy()

            label_enc = LabelEncoder()
            marks = label_enc.fit_transform(marks)
            model = CategoricalHMM(
                n_components=len(set(marks)), n_iter=100, tol=1e-4, verbose=False
            )
            marks = marks.reshape(-1, 1)
            _ = model.fit(marks)
            new_marks = model.sample(gen_len)

            gen_marks.append(label_enc.inverse_transform(new_marks[0].reshape(-1)))

        generated = PredBatch(
            lengths=np.ones(B) * gen_len,
            time=None,
            num_features=None,
            num_features_names=None,
            cat_features={self._feature_name: np.stack(gen_marks)},
        )
        return generated


class BaselineRepeater(BaseGenerator):
    def __init__(self, data_conf: DataConfig):
        super().__init__()
        self.data_conf = data_conf

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)
        hist = deepcopy(hist)
        gen_batch = hist.tail(gen_len)
        if hist.monotonic_time:  # Time is monotonic.
            corr = torch.cat((torch.zeros_like(hist.time[:1]), hist.time))
            corr = corr[hist.lengths - gen_len, torch.arange(hist.time.shape[1])]
            gen_batch.time = gen_batch.time + gen_batch.time[-1] - corr
            # This complicated correction assures same behavior as with timediff
        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch


class BaselineHistSampler(BaseGenerator):
    def __init__(self, data_conf: DataConfig):
        super().__init__()
        self.data_conf = data_conf

    def forward(self, x: GenBatch):
        raise "No need to train a repeator."

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        assert hist.lengths.min() >= gen_len, "Cannot generate when gen_len > hist_len"
        assert isinstance(hist.time, torch.Tensor)

        hist = deepcopy(hist)
        samples = torch.tensor(
            np.array(
                [
                    np.sort(np.random.choice(length, size=gen_len, replace=False))
                    for length in hist.lengths.numpy(force=True)
                ]
            ),
            device=hist.lengths.device,
        ).T  # [gen_len, B]
        gen_batch = replace(
            hist,
            lengths=torch.ones_like(hist.lengths) * gen_len,
            time=gather(hist.time, samples),
            num_features=gather(hist.num_features, samples),
            cat_features=gather(hist.cat_features, samples),
            cat_mask=gather(hist.cat_mask, samples),
            num_mask=gather(hist.num_mask, samples),
        )
        if hist.monotonic_time:  # Time is monotonic.
            corr = torch.cat((torch.zeros_like(hist.time[:1]), hist.time))
            pred_first_time = corr[samples[0], torch.arange(hist.time.shape[1])]
            last_time = hist.time[hist.lengths - 1, torch.arange(hist.time.shape[1])]
            gen_batch.time = gen_batch.time - pred_first_time + last_time
            # This complicated correction assures same behavior as with timediff
        if with_hist:
            hist.append(gen_batch)
            return hist
        else:
            return gen_batch


class Generator(BaseGenerator):
    def __init__(self, data_conf: DataConfig, model_config: ModelConfig):
        super().__init__()

        self.preprocess = create_preprocessor(data_conf, model_config.preprocessor)

        encoder_params = model_config.encoder.params or {}
        encoder_params["input_size"] = self.preprocess.output_dim
        self.encoder = AutoregressiveEncoder(model_config.encoder.name, encoder_params)

        self.projector = Projection(
            self.encoder.output_dim, 2 * self.encoder.output_dim
        )

        self.reconstructor = ReconstructorBase(data_conf, self.projector.output_dim)

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """
        x = self.preprocess(x)  # Sequence of [L, B, D]
        x = self.encoder(x)
        x = self.projector(x)
        x = self.reconstructor(x)
        return x

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        """
        Auto-regressive generation using the transformer

        Args:
            x (Seq): Input sequence [L, B, D]

        """
        hist = deepcopy(hist)

        with torch.no_grad():
            for _ in range(gen_len):
                x = self.preprocess(hist)
                x = self.encoder.generate(x)  # Sequence of shape [1, B, D]
                x = self.projector(x)
                x = self.reconstructor.generate(
                    x
                )  # GenBatch with sizes [1, B, D] for cat, num
                hist.append(x)  # Append GenBatch, result is [L+1, B, D]
        if with_hist:
            return hist  # Return GenBatch of size [L + gen_len, B, D]
        else:
            return hist.tail(gen_len)  # Return GenBatch of size [gen_len, B, D]


class VAE(BaseGenerator):
    def __init__(self, data_conf: DataConfig, model_config: ModelConfig):
        super().__init__()
        self.encoder = VAE_Encoder(
            model_config.vae,
            cat_cardinalities=data_conf.cat_cardinalities,
            num_names=data_conf.num_names,
            batch_transforms=model_config.preprocessor.batch_transforms,
        )

        self.decoder = VAE_Decoder(
            model_config.vae,
            cat_cardinalities=data_conf.cat_cardinalities,
            num_names=data_conf.num_names,
        )

    def forward(self, x: GenBatch) -> PredBatch:
        """
        Forward pass of the Variational AutoEncoder
        Args:
            x (GenBatch): Input sequence [L, B, D]

        """

        assert not self.encoder.pretrained
        x, params = self.encoder(x)
        x = self.decoder(x)
        return x, params

    def generate(self, hist: GenBatch, gen_len: int, with_hist=False) -> GenBatch:
        hist = deepcopy(hist)
        assert hist.target_time.shape[0] == gen_len, hist.target_time.shape
        x = self.encoder(hist.get_target_batch())
        if not self.encoder.pretrained:
            x = x[0]
        x = self.decoder.generate(x)
        if with_hist:
            hist.append(x)
            return hist
        else:
            return x
