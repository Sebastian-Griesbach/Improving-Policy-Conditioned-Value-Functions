import torch
from torch import nn
from typing import Union, List
from tqdm import tqdm

from tc_supervised_learning.data_sampler import PolicyDataSampler
from tc_logging.log_handler import LogHandler

class ActionReconstructionTrainer():
    LOG_TRAIN_LOSS = "train_loss"
    LOG_VALID_LOSS = "valid_loss"

    def __init__(self,
                action_reconstruction_net: nn.Module,
                data_sampler: PolicyDataSampler,
                learning_rate: float,
                train_set_key: str,
                validation_set_key: Union[List[str],str],
                batch_size: int = 256,
                validation_frequency:int = 10,
                log_handler: LogHandler = None,
                device: str = "cpu",
                disable_tqdm: bool = False,
                ) -> None:

        self.action_reconstruction_net = action_reconstruction_net.to(device)
        self.data_sampler = data_sampler

        self.batch_size = batch_size
        self.validation_frequency = validation_frequency

        self.optimizer = torch.optim.Adam(self.action_reconstruction_net.parameters(), lr=learning_rate)

        self.train_set_key = train_set_key
        self.validation_set_keys = validation_set_key if isinstance(validation_set_key, list) else [validation_set_key]
        self.validation_loss_tags = []
        for validation_set_key in self.validation_set_keys:
            self.validation_loss_tags.append(f"{validation_set_key}_loss")


        if(log_handler != None):
            self.log_handler = log_handler
            self.enable_logging = True
        else:
            self.log_handler = None
            self.enable_logging = False


        self.disable_tqdm = disable_tqdm
        self.device = device

        self.loss = nn.MSELoss()
        self.total_train_steps = 0

    def train(self, num_train_steps, log_analysis = True):
        for train_step in tqdm(range(num_train_steps), disable=self.disable_tqdm):
            train_parameters_batch, train_multi_state_batch, train_multi_action_batch = self.data_sampler.sample_data(self.train_set_key, batch_size=self.batch_size)
            train_loss = self.update(train_parameters_batch, train_multi_state_batch, train_multi_action_batch)

            if(self.enable_logging):
                self.log_handler.log_data(data_dict={self.LOG_TRAIN_LOSS: train_loss}, step=self.total_train_steps)

            if(train_step % self.validation_frequency == 0):
                for i, valid_set in enumerate(self.validation_set_keys):
                    valid_parameters_batch, valid_multi_state_batch, valid_multi_action_batch = self.data_sampler.sample_data(valid_set, batch_size=self.batch_size)
                    valid_loss = self.validate(valid_parameters_batch, valid_multi_state_batch, valid_multi_action_batch)

                    if(self.enable_logging):
                        self.log_handler.log_data(data_dict={self.validation_loss_tags[i]: valid_loss}, step=self.total_train_steps)

            self.total_train_steps += 1

        if(log_analysis):
            self._log_analysis()

        if(self.enable_logging):
            self.log_handler.flush_logger()

    def update(self, parameters, states, actions):
        predictions = self.action_reconstruction_net(parameters, states)
        
        self.optimizer.zero_grad()
        loss = self.loss(predictions, actions)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, parameters, states, actions):
        with torch.no_grad():
            predictions = self.action_reconstruction_net(parameters, states)
            loss = self.loss(predictions, actions)
        
        return loss.item()

    def _measure_embed_impact(self, mean_batch = 10000):

        train_parameters_batch, train_multi_state_batch, train_multi_action_batch = self.data_sampler.sample_data(self.train_set_key, batch_size=mean_batch)

        return_dict = {}

        with torch.no_grad():
            train_embeddings = self.action_reconstruction_net.embed(train_parameters_batch)
            train_embeddings_mean = train_embeddings.mean(dim=0, keepdim=True).repeat(mean_batch,1)

            train_mean_embed_predictions = self.action_reconstruction_net.reconstruct(train_embeddings_mean, train_multi_state_batch)
            train_true_embed_predictions = self.action_reconstruction_net.reconstruct(train_embeddings, train_multi_state_batch)

            return_dict["train_mean_embed_loss"] = self.loss(train_multi_action_batch, train_mean_embed_predictions).item()
            return_dict["train_true_embed_loss"] = self.loss(train_multi_action_batch, train_true_embed_predictions).item()

            for i, valid_set in enumerate(self.validation_set_keys):
                valid_parameters_batch, valid_multi_state_batch, valid_multi_action_batch = self.data_sampler.sample_data(valid_set, batch_size=mean_batch)

                valid_embeddings = self.action_reconstruction_net.embed(valid_parameters_batch)
                valid_embeddings_mean = valid_embeddings.mean(dim=0, keepdim=True).repeat(mean_batch,1)

                valid_mean_embed_predictions = self.action_reconstruction_net.reconstruct(valid_embeddings_mean, valid_multi_state_batch)
                valid_true_embed_predictions = self.action_reconstruction_net.reconstruct(valid_embeddings, valid_multi_state_batch)

                return_dict[f"{self.validation_set_keys[i]}_mean_embed_loss"] = self.loss(valid_multi_action_batch, valid_mean_embed_predictions).item()
                return_dict[f"{self.validation_set_keys[i]}_true_embed_loss"] = self.loss(valid_multi_action_batch, valid_true_embed_predictions).item()

        return return_dict

    def _log_analysis(self):

        embed_impact_dict = self._measure_embed_impact()

        log_dict = {"embed_size": self.action_reconstruction_net.embedding_net.embedding_size,
                        "encoder_num_parameters": self._count_parameters(self.action_reconstruction_net.embedding_net),
                        "decoder_num_parameters": self._count_parameters(self.action_reconstruction_net.decoder_net),
                        **embed_impact_dict
                        }

        self.log_handler.log_data(data_dict=log_dict, step=self.total_train_steps)

    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)