from .ar import AR

NUM_TOKENS = 512


def get_model(name):
    name = name.lower()

    if name.startswith("ar"):
        Model = AR
    else:
        raise ValueError("Model name should start with AR or NAR.")

    if "-quarter" in name:
        print("QUARTER")
        model = Model(
            NUM_TOKENS,
            d_model=256,
            n_heads=4,
            n_layers=12,
        )
    elif "-half" in name:
        model = Model(
            NUM_TOKENS,
            d_model=512,
            n_heads=8,
            n_layers=12,
        )
    else:
        if name not in ["ar"]:
            raise NotImplementedError(name)

        model = Model(
            NUM_TOKENS,
            d_model=1024,
            n_heads=8,
            n_layers=18,
        )

    return model