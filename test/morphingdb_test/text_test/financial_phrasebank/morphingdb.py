import torch

def tensor_to_mvec(tensor):
    if not torch.is_tensor(tensor):
         raise ValueError("Input must be a torch.Tensor")
    mvec_data_str = str(tensor.view(-1).tolist())
    mvec_shape = list(tensor.shape)
    mvec_shape_str = str(mvec_shape).replace('[', '{').replace(']', '}')
    mvec_str = mvec_data_str + mvec_shape_str
    return mvec_str


