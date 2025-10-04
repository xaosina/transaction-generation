from .data.data_types import GenBatch
from copy import deepcopy
from typing import Literal

GenerationSetupType = Literal['identity', 'forecast']

class GenSetupBatchProcessor:

    def __init__(self):
        pass

    def on_input(self, batch: GenBatch) -> GenBatch:
        return deepcopy(batch)
    
    def on_generated(self, batch: GenBatch) -> GenBatch:
        return deepcopy(batch)

class ForecastGenSetupBatchProcessor(GenSetupBatchProcessor):
    pass

class IdentityGenSetupBatchProcessor(GenSetupBatchProcessor):

    def on_input(self, batch):
        batch = deepcopy(batch)
        tgt_batch = batch.get_target_batch()
        batch.append(tgt_batch)
        return batch
    
    def on_generated(self, batch):
        # TODO
        return deepcopy(batch)