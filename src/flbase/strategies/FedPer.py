from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass

from .FedAvg import FedAvgClient, FedAvgServer
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch


class FedPerClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)


class FedPerServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        # FedPer do not aggregate head as implemented in the FedBaBu's code
        # head_key = [name for name, p in self.server_side_client.model.named_parameters() if 'fc' in name or 'prototype' in name]
        head_key = [name for name, p in self.server_side_client.model.named_parameters() if 'prototype' in name]
        self.exclude_layer_keys.update(head_key)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedPerServer: the following keys will not be aggregate:\n ", self.exclude_layer_keys)
        freeze_layers = []
        for param in self.server_side_client.model.named_parameters():
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])
        if len(freeze_layers) > 0:
            print("FedPerServer: the following layers will not be updated:", freeze_layers)
        # print('FedPer line62 (init time) sever model prototype', self.server_model_state_dict['prototype'][0, :3])

    # def testing(self, round, active_only=True, **kwargs):
    #     """
    #     active_only: only compute statiscs with to the active clients only
    #     """
    #     # get the latest global model
    #     # the excluded layers will never be updated. Set
    #     #  exclude_keys = set() so that the exclude part on the server
    #     # side client model is consistent with the initial point
    #     print('before prototype', self.server_side_client.model.prototype[0, :3])
    #     self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
    #     print('after prototype', self.server_side_client.model.prototype[0, :3])
    #     # print('FedPer line72: (testing time) sever model prototype', self.server_model_state_dict['prototype'][0, :3])
    #     # test the performance for global models

    #     self.server_side_client.testing(round, testloader=None)  # use global testdataset
    #     print('FedPer remove return line 77')
    #     return
    #     client_indices = self.clients_dict.keys()
    #     if active_only:
    #         client_indices = self.active_clients_indicies
    #     for cid in client_indices:
    #         client = self.clients_dict[cid]
    #         # test local model on the global testset
    #         client.testing(round, self.server_side_client.testloader)
