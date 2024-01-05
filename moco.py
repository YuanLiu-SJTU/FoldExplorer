#modified from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch
from encoder import GATModel


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, in_channels, graph_out_channels, esm_out_channels,
                 dim, device, K=1024, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = GATModel(in_channels, graph_out_channels, esm_out_channels)
        self.encoder_k = GATModel(in_channels, graph_out_channels, esm_out_channels)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        
        self.queue[:, ptr: ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, data, test=False):
        data1 = data2 = None
        if isinstance(data, list) and ~test:
            data1, data2 = data
            q = self.encoder_q(data1)
            q = F.normalize(q, dim=1)
        elif isinstance(data, torch_geometric.data.batch.Data) and test:
            q = self.encoder_q(data)
            return F.normalize(q, dim=1)
        else:
            raise RuntimeError("Unknown type of network input! Please check it!")

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(data2)
            k = F.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1).to(self.device)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels
