import torch
from torch.distributions import Categorical
import torch.nn.functional as F
# from tabsyn.transformer.ar import AR

def compute_celoss(h_list, y_list, ignore_index):
    count = 0
    total_loss = 0.
    for i in range(y_list.__len__()):
        for j in range(y_list[i].shape[0]):
            if not (y_list[i][j] == ignore_index).all():
                loss = F.cross_entropy(
                        h_list[i][j],
                        y_list[i][j],
                        ignore_index=ignore_index
                    )
                count += 1
                total_loss += loss
    return total_loss / count

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


# ======= Адаптеры для входов =======
class BaseInputAdapter:
    def prepare(self, model, text_list, proms_list, resps_list):
        """Преобразует входные данные в общий формат (x_list)."""
        raise NotImplementedError


class CodesInputAdapter(BaseInputAdapter):
    def prepare(self, model, text_list, proms_list, resps_list):
        # В режиме 'codes' выбор эмбеддингов зависит от флага on_codes
        if model.on_codes:
            return model._samplewise_merge_tensors(
                # model.text_emb(text_list),
                model.proms_emb(proms_list),
                model.resps_emb(resps_list),
                sep=model.sep
            )
        else:
            return model._samplewise_merge_tensors(
                # model.text_emb(text_list),
                model.linear_proj(proms_list),
                model.linear_proj(resps_list),
                sep=model.sep
            )


class NumInputAdapter(BaseInputAdapter):
    """
    Adapter for inputs:
    proms_list: B x L x 48
    resps_list: B x L x 48
    """
    def prepare(self, model, text_list, train_seq):
        return model._samplewise_merge_tensors(
            torch.zeros(text_list.shape + torch.Size([model.d_model]), device=train_seq.device),
            model.linear_proj(train_seq),
            sep=model.sep
        )

class NoSepInputAdapter(BaseInputAdapter):
    """
    Adapter for inputs:
    proms_list: B x L x 48
    resps_list: B x L x 48
    """
    def prepare(self, model, _, train_seq):
        return model.linear_proj(train_seq)



# ======= Адаптеры для выходов =======
class BaseOutputAdapter:
    def __init__(self, loss_type="full"):
        self.loss_type = loss_type

    def compute_logits(self):
        """Вычисляет логиты (выходы классификатора) и возвращает список h_list."""
        raise NotImplementedError
    
    def after_attention(self):
        raise NotImplementedError
    
    def compute_loss(self):
        """Если таргеты переданы – считает loss, иначе возвращает None."""
        raise NotImplementedError

    def sample(self):
        """Сэмплирует итоговый ответ из логитов."""
        raise NotImplementedError
    
    @staticmethod
    def validate_target_list(targ_list):
        """Проверяет, что targ_list не пуст и не содержит пустых элементов."""
        if targ_list is None or any(len(t) == 0 for t in targ_list):
            raise ValueError("Невозможно вычислить потери с пустым targ_list.")
    
    @staticmethod
    def create_ignore_tensor(size, ignore_index, device):
        """Создает тензор игнорирования для промтов."""
        return torch.full(size, ignore_index, device=device)
    
    @staticmethod
    def create_prom_list(proms_list, ignore_index):
        """Создает список промтов, заполняя их значением ignore_index."""
        return torch.full_like(proms_list, ignore_index)
        # return [torch.full_like(t, ignore_index) for t in proms_list]
    
    @staticmethod
    def shift_tensor(tensor, ignore_index):
        """Сдвигает тензор на одну позицию и заполняет последнюю позицию значением ignore_index."""
        tensor = tensor.roll(-1, dims=1)
        tensor[:, -1] = ignore_index
        return tensor


class CodesOutputAdapter(BaseOutputAdapter):
    def compute_logits(self, model, x, m, x_list):
        if model.n_resp_levels > 1:
            h = [model.classifier[i](x) * m for i in range(model.n_resp_levels)]
        else:
            h = [model.classifier(x) * m]
        # Если on_codes или cat_loss – применяем обрезание с перестановкой размерностей
        if model.on_codes or model.cat_loss:
            h_list = [
                torch.stack([hi[:li] for hi, li in zip(hk, map(len, x_list))], dim=0)
                .permute(1, 0, 2)
                for hk in h
            ]
        else:
            h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]
        return h_list

    def compute_loss(self, model, h_list, text_list, proms_list, targ_list,
                     cat_targ_list, cat_proms_list, shift_targ_list):
        # В режиме codes используем targ_list (и игнорируем cat_*)
        if targ_list is None:
            return None
        if any(len(t) == 0 for t in targ_list):
            raise ValueError("Cannot compute loss given empty targ_list.")
        device = h_list[0].device
        ignore_sep = torch.full((proms_list[0].shape[1],), model.ignore_index, device=device)
        prom_list = [torch.full_like(t, model.ignore_index) for t in proms_list]
        assert text_list[0].shape == torch.Size([0]), 'Are you passed the text here???'
        text_prom_list = model._samplewise_merge_tensors(
            [t.reshape(0, proms_list[0].shape[1]) for t in text_list],
            prom_list, sep=ignore_sep
        )
        for i in range(len(text_prom_list)):
            if model.resp_loss_only:
                text_prom_list[i][:] = model.ignore_index
            else:
                text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                text_prom_list[i][-1] = model.ignore_index
        if shift_targ_list:
            targ_list = [t.clone() for t in targ_list]
            for i in range(len(targ_list)):
                targ_list[i] = targ_list[i].roll(-1, dims=0)
                targ_list[i][-1] = model.ignore_index
        y_list = model._samplewise_merge_tensors(text_prom_list, targ_list, sep=ignore_sep)
        # Пример: можно легко переключаться между compute_celoss и compute_mseloss
        return {'nll': compute_celoss(h_list, y_list, model.ignore_index)}

    def sample(self, model, h_list, resps_list, return_all_resp, sampling_temperature):
        if return_all_resp:
            logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
            ret = [Categorical(logits=logits_i / sampling_temperature).sample() for logits_i in logits]
        else:
            if model.on_codes:
                logits = torch.stack([hi[-1] for hi in h_list])
                ret = Categorical(logits=logits / sampling_temperature).sample()
            else:
                ret = torch.stack([hi[-1] for hi in h_list])
        return ret

class CatOutputAdapter(BaseOutputAdapter):
    def __init__(self, c_dim, c_number, loss_type="full"):
        super().__init__(loss_type)
        self.c_dim = c_dim
        self.c_number = c_number

    def compute_logits(self, model, x, m):
        return model.classifier(x) * m
    
    def after_attention(self, model, x, m):
        return model.last_linear(x) * m

    def compute_loss(self, y_pred, y_true, ignore_index):
        self.validate_target_list(y_true)
        # assert y_pred.shape[-2] == (self.c_number)

        match self.loss_type:
            case "full":
                y_true = self.shift_tensor(y_true, ignore_index)
                return {'nll': F.cross_entropy(
                    y_pred[:, :-1, ...].reshape(-1, self.c_number, self.c_dim).permute(0, 2, 1), 
                    y_true[:, :-1, ...].reshape(-1, self.c_number).long()
                    )}
            # F.cross_entropy(y_pred.reshape(-1, 32, 32).permute(0, 2, 1).reshape(-1, 32), y_true.reshape(-1))
            case "last":
                return {'nll': F.cross_entropy(
                    y_pred[:, -1, ...].reshape(-1, self.c_number, self.c_dim).permute(0, 2, 1), 
                    y_true[:, ...].reshape(-1, self.c_number).long()
                    )}

    def sample(self, h_list, return_all_resp, sampling_temperature):
        if return_all_resp:
            raise NotImplementedError
            # logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
            # ret = [Categorical(logits=logits_i / sampling_temperature).sample() for logits_i in logits]
        else:
            logits = h_list[:, -1, :]
            ret = Categorical(logits=logits.view(-1, self.c_number, self.c_dim) / sampling_temperature).sample()
        return ret


class Num16x48OutputAdapter(BaseOutputAdapter):
    def __init__(self, loss_type="full"):
        super().__init__(loss_type)

    def compute_logits(self, model, x, m):
        # В режиме cat сначала применяем last_linear, затем классификаторы
        h = [clf(x) * m for clf in model.classifier]
        h_list = torch.stack(h).permute(1, 2, 0, 3)
        return h_list


    def after_attention(self, model, x, m):
        ret = model.last_linear(x) * m
        return ret

    def compute_loss(self, y_pred, y_true, ignore_index):
        self.validate_target_list(y_true)

        match self.loss_type:
            case "full":
                y_true = self.shift_tensor(y_true, ignore_index)

                return {'nll': F.mse_loss(
                    y_pred[:, :-1, ...].reshape(-1, *y_pred.shape[2:]), 
                    y_true[:, :-1, ...].reshape(-1, *y_true.shape[2:])
                    )}
            case "last":
                return {'nll': F.mse_loss(
                    y_pred[:, -1, ...].reshape(-1, *y_pred.shape[2:]), 
                    y_true.reshape(-1, *y_true.shape[2:])
                    )}

    def sample(self, h_list, return_all_resp, sampling_temperature):
        assert not return_all_resp, "Something wring with returns in sample"
        return h_list[:, -1, ...]
    

class Num48OutputAdapter(BaseOutputAdapter):
    def __init__(self, loss_type="full"):
        super().__init__(loss_type)

    def compute_logits(self, model, x, m):
        h = model.classifier(x) * m 
        return h

    def after_attention(self, model, x, m):
        ret = model.last_linear(x) * m
        return ret

    def compute_loss(self, y_pred, y_true, ignore_index):
        self.validate_target_list(y_true)
        # import numpy as np
        # np.save('here-1.npy', y_true.detach().cpu().numpy())
        match self.loss_type:
            case "full":
                y_true = self.shift_tensor(y_true, ignore_index=ignore_index)
                return {'nll': F.mse_loss(y_pred[:, :-1, :], y_true[:, :-1, :])}
            case "last":
                return {'nll': F.mse_loss(y_pred[:, -1, :], y_true[:, -1, :])}
        # np.save('here-2.npy', y_true.detach().cpu().numpy())
        # return {'nll': compute_mseloss(h_list, y_list, model.ignore_index)}
        # return {'nll': F.mse_loss(h_list[:, -targ_list.shape[1]:], targ_list)}

    def sample(self, h_list, return_all_resp, sampling_temperature):
        assert not return_all_resp, "Something wring with returns in sample"
        if return_all_resp:
            raise NotImplementedError
        else:
            ret = h_list[:, -1]
        return ret