from copy import copy
from generation.data.batch_tfs import NewFeatureTransform

def get_features_after_transform(data_conf, batch_transforms, model_config):
    cat_cardinalities = copy(data_conf.cat_cardinalities) or {}
    num_names = copy(data_conf.num_names) or []
    if batch_transforms is not None:
        assert model_config.frozen, "Transforms are only for pretrained models!"
        for tfs in batch_transforms:
            if isinstance(tfs, NewFeatureTransform):
                for num_name in tfs.num_names:
                    num_names += [num_name]
                for cat_name, card in tfs.cat_cardinalities.items():
                    cat_cardinalities[cat_name] = card
                num_names = [n for n in num_names if n not in tfs.num_names_removed]
                cat_cardinalities = {
                    k: v
                    for k, v in cat_cardinalities.items()
                    if k not in tfs.cat_names_removed
                }
    return num_names, cat_cardinalities