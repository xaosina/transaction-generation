from .rnn import RNN

NUM_TOKENS = 512

def get_model(name):
    name = name.lower()

    if name.startswith("gru"):
        Model = RNN
    else:
        raise ValueError("Model name should start with AR or NAR.")

    model = Model(
        input_dim="VAE_HIDDEN_DIM", # TODO Repair
        hidden_dim=512,
        output_dim="VAE_HIDDEN_DIM"
    )

    return model