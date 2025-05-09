from ebes.model import BaseModel
from ebes.model.seq2seq import Projection
from ebes.types import Seq

from ..data.data_types import GenBatch, LatentDataConfig, PredBatch

# from ..data.preprocess.vae.models.vae.model import Decoder_model as VAE_Decoder


class ReconstructorMSE:
    def __init__(self):

        self.decoder = VAE_Decoder  # Import decoder from VAE.
        self.rvqvae = ...  # #Import rvqvae from RVQ-VAE.

    def forward(self, x):
        return x

    def generate(self, x):
        x = self.decoder(x)
        # TODO: Other reconstruction.
        return x


class ReconstructorBase(BaseModel):
    def __init__(self, data_conf: LatentDataConfig, in_features):
        super().__init__()
        self.num_names = data_conf.num_names
        self.cat_cardinalities = data_conf.cat_cardinalities
        out_dim = 1  # Time
        if self.num_names:
            out_dim += len(self.num_names)
        if self.cat_cardinalities:
            out_dim += sum(self.cat_cardinalities.values())

        self.head = Projection(in_features, out_dim)

    def forward(self, x: Seq) -> PredBatch:
        out = self.head(x).tokens
        time = out[:, :, 0]
        start_id = 1

        num_features = None
        if self.num_names:
            num_features = out[:, :, 1 : len(self.num_names) + 1]
            start_id += len(self.num_names)
        cat_features = None
        if self.cat_cardinalities:
            cat_features = {}
            for cat_name, cat_dim in self.cat_cardinalities.items():
                cat_features[cat_name] = out[:, :, start_id : start_id + cat_dim]
                start_id += cat_dim
        assert start_id == out.shape[2]
        return PredBatch(
            lengths=x.lengths,
            time=time,
            num_features=num_features,
            num_features_names=self.num_names,
            cat_features=cat_features,
        )

    def generate(self, x: Seq) -> GenBatch:
        return self.forward(x).to_batch()
