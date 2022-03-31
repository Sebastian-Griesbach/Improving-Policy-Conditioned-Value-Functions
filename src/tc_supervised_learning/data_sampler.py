import torch
import numpy as np
from typing import Dict

class PolicyDataSampler():
    def __init__(self, 
                dataset_path_dict: Dict[str, str],
                use_named_params: bool,
                num_state_action_pairs: int = 1,
                device: str = "cpu",
                id_params: int = 0, 
                id_states: int = 1, 
                id_actions: int = 2) -> None:

        self.data_set_dict = {key:torch.load(path) for key, path in dataset_path_dict.items()}
        self.num_state_action_pairs = num_state_action_pairs

        self.sample_flat_params = not use_named_params
        self.use_named_params = use_named_params

        self.device = device

        self.id_params = id_params
        self.id_states = id_states
        self.id_actions = id_actions

    def sample_data(self, data_set_key, batch_size):
        
        sample_ids = np.random.choice(np.arange(len(self.data_set_dict[data_set_key])), size=batch_size)

        if(self.use_named_params):
            parameters = self._get_named_parameters(self.data_set_dict[data_set_key], sample_ids, self.id_params)
        else:
            parameters = self._get_flat_parameters(self.data_set_dict[data_set_key], sample_ids, self.id_params)

        multi_states, multi_actions = self._get_state_action_pairs(self.data_set_dict[data_set_key], sample_ids, self.num_state_action_pairs, self.id_states, self.id_actions)

        return parameters, multi_states, multi_actions

    def _get_state_action_pairs(self, data_set, sample_ids, num_state_action_pairs, id_states, id_actions):
        multi_states = []
        multi_actions = []

        for idx in sample_ids:
            sub_sample_ids = np.random.choice(np.arange(len(data_set[idx][id_states])), size=num_state_action_pairs)
            multi_states.append(torch.vstack([data_set[idx][id_states][i] for i in sub_sample_ids]).reshape(1,-1))
            multi_actions.append(torch.vstack([data_set[idx][id_actions][i] for i in sub_sample_ids]).reshape(1,-1))

        multi_states = torch.vstack(multi_states).to(self.device)
        multi_actions = torch.vstack(multi_actions).to(self.device)

        return multi_states, multi_actions

    def _get_named_parameters(self, data_set, sample_ids, id_params):
        param_keys = data_set[0][id_params].keys()
        named_params = {key:[] for key in param_keys}

        for idx in sample_ids:
            for key in param_keys: 
                named_params[key].append(data_set[idx][id_params][key].unsqueeze(0))

        for key in param_keys: 
            named_params[key] = torch.vstack(named_params[key]).to(self.device)

        return named_params

    def _get_flat_parameters(self, data_set, sample_ids, id_params):
        flat_params = []

        for idx in sample_ids:
            flat_params.append(torch.hstack([param.data.flatten() for param in data_set[idx][id_params].values()]).flatten())
        
        flat_params = torch.vstack(flat_params).to(self.device)

        return flat_params
    