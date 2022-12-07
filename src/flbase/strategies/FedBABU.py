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


class FedBABUClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)

    def to_freeze_layers(self, layers_to_freeze):
        '''
        `layers_to_freeze` indicates layers to freeze and unfreeze unindicated layers.
        '''
        for name, p in self.model.named_parameters():
            try:
                if name in layers_to_freeze:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            except:
                pass


class FedBABUServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)

        # FedBABU do not aggregate head and do not update head
        head_key = [name for name in self.server_side_client.model.state_dict().keys() if 'prototype' in name]
        self.exclude_layer_keys.update(head_key)
        self.include_layer_keys = set([key for key in self.server_model_state_dict
                                       if key not in self.exclude_layer_keys])

        if len(self.exclude_layer_keys) > 0:
            print(f"FedBABUServer: the following keys will not be aggregate:\n ", self.exclude_layer_keys)

        # freeze heads
        self.server_side_client.to_freeze_layers(self.exclude_layer_keys)
        # all clients use same heads and freeze
        for cid in self.clients_dict.keys():
            client = self.clients_dict[cid]
            client.model.prototype = deepcopy(self.server_side_client.model.prototype)
            client.to_freeze_layers(self.exclude_layer_keys)
            # print([(name, p.requires_grad) for name, p in client.model.named_parameters()])
            # print(client.model.state_dict()['prototype.weight'][0,:5])

    def run(self, **kwargs):
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            setup_seed(r)
            selected_indices = self.select_clients(self.server_config['participate_ratio'])
            if self.server_config['drop_ratio'] > 0:
                # mimic the stragler issues; simply drop them
                self.active_clients_indicies = np.random.choice(selected_indices, int(
                    len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
            else:
                self.active_clients_indicies = selected_indices
            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            client_uploads = []
            train_start = time.time()
            # TODO fix body, train head
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.training(r, client.client_config['num_epochs'])
                client_uploads.append(client.upload())

            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")

            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            # agg_start = time.time()
            # tmp = self.exclude_layer_keys # test for agg all layer
            # self.exclude_layer_keys = set() # test for agg all layer

            self.aggregate(client_uploads, round=r)

            # self.exclude_layer_keys = tmp # test for agg all layer
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", self.gfl_test_acc_dict[r]['acc_by_criteria'])
                print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])
                if len(self.gfl_test_acc_dict) >= 2:
                    current_key = r
                    if self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform'] > best_test_acc:
                        best_test_acc = self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform']
                        self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                        tqdm.write(f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                        if 'filename' in kwargs:
                            torch.save(self.server_model_state_dict_best_so_far, kwargs['filename'])
                else:
                    best_test_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                         }

                for criteria in self.average_pfl_test_acc_dict[r].keys():
                    stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)
            # sanity check
            # for cid in self.clients_dict.keys():
            #     client = self.clients_dict[cid]
            #     print(client.model.state_dict()['prototype.weight'][0, :5])
        # finetune heads for clients
        final_round = self.server_config['num_rounds'] + 1

        for cid in self.clients_dict.keys():
            client = self.clients_dict[cid]
            # unfreeze all
            client.to_freeze_layers(set())
            # print([(name, p.requires_grad) for name, p in client.model.named_parameters()])
            client.training(final_round, client.client_config['FedBABU_finetune_epoch'])
            # print(client.model.state_dict()['prototype.weight'][0, :5])
            client.testing(final_round, self.server_side_client.testloader)
        self.server_side_client.testing(final_round, self.server_side_client.testloader)
        self.collect_stats(stage="test", round=final_round, active_only=False)
        print("After fine-tuning...")
        print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[final_round])
        if kwargs['use_wandb']:
            for criteria in self.average_pfl_test_acc_dict[final_round].keys():
                stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[final_round][criteria]
            wandb.log(stats)
