from dataclasses import dataclass, field
from .reconstructors import ReconstructorMSE
from .projectors import ProjectMSE
from .encoders import ARTransformer
from ebes.model import BaseModel
from ..data.types import Seq


# TODO: Think
@dataclass
class Generator:
    preprocess: PreprocessConfig = field(default=None)
    encoder: PreprocessConfig = field(default=None)
    projector: ProjectorConfig = field(default=None)
    reconstructor: PreprocessConfig = field(default=None)

class Generator(BaseModel):
    def __init__(self,):

        self.preprocess = ...

        self.encoder = ARTransformer()
        
        self.projector = ProjectMSE()

        self.reconstructor = ReconstructorMSE()
        
    def forward(self, x: Seq):

        x = self.preprocess(x)

        x = self.encoder(x)
        
        x = self.projector(x)
        
        x = self.reconstructor(x)

        return x

    def generate(self, x: Seq):

        x = self.preprocess(x)
        
        ret = self.encoder(x)
        
        ret = self.projector(ret)
        
        ret = self.reconstructor(ret)

        return ret
