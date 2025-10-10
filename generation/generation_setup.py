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

def get_gensetup_batch_processor(
        gensetup_type: GenerationSetupType, 
        *args, 
        **kwargs,
    ) -> GenSetupBatchProcessor:

    if gensetup_type == 'forecast':
        return ForecastGenSetupBatchProcessor()
    
    if gensetup_type == 'identity':
        return IdentityGenSetupBatchProcessor()
    
    raise Exception(f"Unsupported gensetup type '{gensetup_type}'!")


class ForecastGenSetupBatchProcessor(GenSetupBatchProcessor):
    pass


class IdentityGenSetupBatchProcessor(GenSetupBatchProcessor):

    def on_input(self, batch):
        batch = deepcopy(batch)
        tgt_batch = batch.get_target_batch()
        batch.append(tgt_batch)
        return batch
    
    def on_generated(self, batch):
        return deepcopy(batch)