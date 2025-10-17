from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import logging
logger = logging.getLogger(__name__)

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

class CosineWithWarmUp(LambdaLR):
    
    def __init__(
        self, 
        optimizer, 
        warmup=4,
        total_steps=100,
    ):
        def lr_lambda(step):
            if step < warmup:
                res = float(step + 1) / float(max(1, warmup))
                logger.info(f'CosineWithWarmUp lr setted to {res}')
                return res
            progress = float(step + 1 - warmup) / float(max(1, total_steps - warmup))
            res = 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine to 0
            logger.info(f'CosineWithWarmUp lr setted to {res}')
            return res

        super().__init__(
            optimizer, 
            lr_lambda,
        )
    
    def step(self, epoch=None, loss=None, metrics=None):
        super().step(epoch=epoch)