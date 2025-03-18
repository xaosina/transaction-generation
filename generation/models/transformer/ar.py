
from .base import Base

class AR(Base):
    
    @property
    def n_resp_levels(self):
        return 1

    @property
    def casual(self):
        return True

    @property
    def use_stop_token(self):
        return False

    @property
    def norm_type(self):
        return "ln"

    @property
    def resp_loss_only(self):
        return False

    def forward(self, x):
        return super().forward(
                x,
                quant_levels=None,
                shift_targ_list=False,
                return_all_resp=False,
                )
