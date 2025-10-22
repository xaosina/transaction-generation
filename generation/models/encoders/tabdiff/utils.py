import torch


def permute_dict_tensor(tensor_dict,permutation_order=(1,0,2)):


    for key,tensor in tensor_dict.items():

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == len(permutation_order)

        tensor_dict[key] = tensor.permute(permutation_order)

    return tensor_dict