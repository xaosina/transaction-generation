from ebes.model.preprocess import Batch2Seq

from ..data.data_types import DataConfig


def create_preprocessor(
    data_conf: DataConfig, cat_emb_dim, num_emb_dim, time_process, num_norm
):
    return Batch2Seq(
        cat_cardinalities=data_conf.cat_cardinalities,
        num_features=data_conf.num_names,
        cat_emb_dim=cat_emb_dim,
        num_emb_dim=num_emb_dim,
        time_process=time_process,
        num_norm=num_norm,
    )
