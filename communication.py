import torch


def update_previous_state(current_state: dict, previous_state: dict) -> None:
    """Updates previous state to current state."""
    for (key_1, weights_1), key_2 in zip(current_state.items(),
                                         previous_state.keys()):
        assert key_1 == key_2
        previous_state[key_2] = weights_1.clone().detach()


def sample_from_models(model: torch.nn.Module, state_dict_1: dict,
                       state_dict_2: dict, state_2_prob: float) -> None:
    """Updates model with a weighted random sample of state_dict 1 and 2."""
    sampled_state_dict = {}
    for (key_1, weights_1), (key_2, weights_2) in zip(state_dict_1.items(),
                                                      state_dict_2.items()):
        assert key_1 == key_2
        mask = torch.rand(weights_1.shape, device=weights_1.device)
        mask = mask > state_2_prob
        sample = torch.where(mask, weights_1, weights_2)
        sampled_state_dict[key_1] = sample
    model.load_state_dict(sampled_state_dict)
