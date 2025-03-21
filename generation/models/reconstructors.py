from ..data.preprocess.vae.models.vae.model import Decoder_model as VAE_Decoder
from ebes.model.preprocess import Batch2Seq
from data.types import Batch

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


class ReconstructorBaseline: 
    def __init__(self):
        pass

    def forward(self, x):
        ...

    def generate(self, x):
        ...