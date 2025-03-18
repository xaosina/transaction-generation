from ..data.preprocess.vae.models.vae.model import Decoder_model as VAE_Decoder


class ReconstructorMSE:
    def __init__(self):
        
        self.decoder = VAE_Decoder # Import decoder from VAE.
        self.rvqvae = ... # #Import rvqvae from RVQ-VAE. 

    def forward(self, x):
        return x

    def generate(self, x):
        x = self.decoder(x)
        # TODO: Other reconstruction.
        return x

