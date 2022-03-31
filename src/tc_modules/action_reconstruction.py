import torch

from tc_modules.mlp_embedding import MLPEmbeddingNetwork
from tc_modules.embedding_decoder import PolicyDecoderNet

class PolicyReconstructor(torch.nn.Module):
    def __init__(self, 
                embedding_net: MLPEmbeddingNetwork,
                decoder_net: PolicyDecoderNet):

        super(PolicyReconstructor, self).__init__()

        self.embedding_net = embedding_net
        self.decoder_net = decoder_net

    def embed(self, parameters):
        return self.embedding_net(parameters).flatten(start_dim=1)

    def reconstruct(self, embeddings, states):
        return self.decoder_net(embeddings, states)
    
    def forward(self, parameters, states):
        embedding = self.embed(parameters)
        return self.reconstruct(embedding, states)