from dataclasses import dataclass, field
from .reconstructors import ReconstructorMSE
from .projectors import ProjectMSE
from .encoders import ARTransformer
from ebes.model import BaseModel
from ..data.types import Seq


# # TODO: Think
# @dataclass
# class Generator:
#     preprocess: PreprocessConfig = field(default=None)
#     encoder: PreprocessConfig = field(default=None)
#     projector: ProjectorConfig = field(default=None)
#     reconstructor: PreprocessConfig = field(default=None)

class Generator(BaseModel):
    def __init__(self,):

        self.preprocess = ...

        self.encoder = ARTransformer()

        self.projector = ProjectMSE()

        self.reconstructor = ReconstructorMSE()
        
    def forward(self, x: Seq): 
        """
        Forward pass of the Auto-regressive Transformer
        Args:
            x (Seq): Input sequence [L, B, D]

        """
        x = self.preprocess(x) # B, L, D

        x = self.encoder(x)
        
        x = self.projector(x)
        
        x = self.reconstructor(x)

        return x

    def generate(self, x: Seq):
        """
        Auto-regressive generation using the transformer
        
        Args:
            x (Seq): Input sequence [L, B, D]
            
        """
        x = self.preprocess(x)
        
        ret = self.encoder.generate(x)
        
        ret = self.projector.generate(ret)
        
        ret = self.reconstructor.generate(ret)

        return ret
