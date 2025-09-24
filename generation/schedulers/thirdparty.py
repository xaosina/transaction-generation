from torch.optim.lr_scheduler import ReduceLROnPlateau

class ReduceLROnPlateauScheduler(ReduceLROnPlateau):
    
    def __init__(
        self, 
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=10, 
        threshold=0.0001, 
        threshold_mode='rel', 
        cooldown=0, 
        min_lr=0, 
        eps=1e-08
    ):
        super().__init__(
            optimizer, 
            mode=mode, 
            factor=factor, 
            patience=patience, 
            threshold=threshold, 
            threshold_mode=threshold_mode, 
            cooldown=cooldown, 
            min_lr=min_lr, 
            eps=eps
        )
    
    def step(self, epoch=None, loss=None, metrics=None):
        assert loss is not None
        super().step(loss)