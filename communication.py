import torch
from typing import Callable


def get_model_snapshot(model: torch.nn.Module) -> dict:
    """Returns a cloned detached model state dict."""
    state_dict = {}
    for key, weights in model.state_dict().items():
        state_dict[key] = weights.clone().detach()
    return state_dict


def update_previous_state(current_state: dict, previous_state: dict) -> None:
    """Updates previous state to current state."""
    for (key_1, weights_1), key_2 in zip(current_state.items(),
                                         previous_state.keys()):
        assert key_1 == key_2
        previous_state[key_2] = weights_1.clone().detach()


def sample_from_models(
        model: torch.nn.Module, state_dict_1: dict, state_dict_2: dict,
        mask_func: Callable[[torch.Tensor], torch.Tensor], sim_rank: int
) -> None:
    """Updates model with a weighted random sample of state_dict 1 and 2."""
    sampled_state_dict = {}
    for (key_1, weights_1), (key_2, weights_2) in zip(state_dict_1.items(),
                                                      state_dict_2.items()):
        assert key_1 == key_2
        mask = torch.rand(weights_1.shape, device=weights_1.device)
        mask = mask_func(mask, sim_rank=sim_rank)
        sample = torch.where(mask, weights_1, weights_2)
        sampled_state_dict[key_1] = sample
    model.load_state_dict(sampled_state_dict)


def uniform_masking(masking_prob: float, num_workers: int):
    def masking_func(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x > masking_prob
    return masking_func


def uniform_structred_masking(masking_prob: float, num_workers: int):
    calibrated_prob = masking_prob / (1 - 1 / num_workers)

    def masking_func(x: torch.Tensor, **kwargs) -> torch.Tensor:
        chunk_size = x.shape[0] // num_workers
        sim_rank = kwargs.pop('sim_rank', None)
        if sim_rank is None:
            raise KeyError(
                'sim_rank must be provided when calling structured_masking')
        mask = x > calibrated_prob
        if x.dim() == 2:
            mask[sim_rank * chunk_size: (sim_rank + 1) * chunk_size, :] = True
        elif x.dim() == 1:
            mask[sim_rank * chunk_size: (sim_rank + 1) * chunk_size] = True
        else:
            raise RuntimeError('x has more than 2 dimensions')
        return mask
    return masking_func


def get_masking_func(
        masking_type: str, masking_prob, num_workers
) -> Callable[[torch.Tensor], torch.Tensor]:
    return {
        'uniform': uniform_masking,
        'structured_uniform': uniform_structred_masking
    }[masking_type](masking_prob, num_workers)
