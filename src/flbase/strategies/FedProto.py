from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..model import ModelWrapper
from ..models.CNN import *
from ..models.MLP import *
from ..strategies.FedAvg import FedAvgServer
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205


def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221


def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


class FedProtoClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)

        self._initialize_model()
        self.local_protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.is_on_server = False

    def set_on_server(self):
        self.is_on_server = True

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        g_head = deepcopy(self.model.prototype)
        self.model.prototype = nn.Identity()
        self.model = ModelWrapper(self.model, g_head, self.model.config)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)

    def training(self, round, num_epochs):
        setup_seed(round + self.client_config['global_seed'])
        # train mode
        self.model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        optimizer = setup_optimizer(self.model, self.client_config, round)

        for i in range(num_epochs):
            # accumulate protos
            local_embedding_by_class = defaultdict(list)
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                embedding, out_g = self.model(x, return_embedding=True)
                # cross entropy loss
                loss = self.criterion(out_g, y)
                if self.global_protos != None:
                    # compute the regularization term
                    # the implementation takes from https://github.com/yuetan031/fedproto/blob/97e42e2741e1ae2e2fce2465e80f38f6997f8836/lib/update.py#L165
                    # however this is not striaghtly follows the paper
                    # it computes avg(||phi_i - C_i||^2),
                    # where phi_i is embedding and c_i is server side prototye for class i
                    place_hldr = torch.zeros_like(embedding)
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        place_hldr[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(place_hldr, embedding) * self.client_config['FedProto_lambda']
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()

                # compute local prototyes
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    local_embedding_by_class[y_c].append(embedding[i, :].detach().data)

                predicted = out_g.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

        # the original impl aggregate at the server side... I move to the client side
        # https://github.com/yuetan031/fedproto/blob/97e42e2741e1ae2e2fce2465e80f38f6997f8836/exps/federated_main.py#L102
        self.local_protos = agg_func(local_embedding_by_class)

    def upload(self):
        return self.local_protos

    def set_protos(self, global_protos):
        self.global_protos = deepcopy(global_protos)

    def testing(self, round, testloader=None):
        self.model.eval()
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        # all_classes_sorted = sorted(test_count_per_class.keys())
        # test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in all_classes_sorted])
        # num_classes = len(all_classes_sorted)
        num_classes = self.client_config['num_classes']
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in range(num_classes)])
        test_correct_per_class = torch.tensor([0] * num_classes)
        test_correct_per_class_p = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # prediction use local model
                embedding, out_p = self.model(x, return_embedding=True)
                predicted_p = out_p.data.max(1)[1]
                # use global prototype
                # dist = float('inf') * torch.ones(y.shape[0], self.client_config['num_classes']).to(self.device)
                # for i, r in enumerate(embedding):
                #     for j, pro in self.global_protos.items():
                #         dist[i, j] = self.loss_mse(r, pro)
                targets = float('inf') * torch.ones((self.client_config['num_classes'], self.global_protos[0].shape[0])).to(self.device)
                for j, pro in self.global_protos.items():
                    targets[j] = pro
                # targets = torch.vstack([v for v in self.global_protos.values()])
                dist = torch.cdist(embedding, targets, p=2.0)
                # prediction use global prototyoe
                predicted_g = torch.argmin(dist, dim=1)
                # stats
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted_g == y) * (y == cls)).sum().item()
                    test_correct_per_class_p[cls] += ((predicted_p == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        acc_by_critertia_dict_p = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

            acc_by_critertia_dict_p[k] = (((weight_per_class_dict[k] * test_correct_per_class_p).sum()) /
                                          ((weight_per_class_dict[k] * test_count_per_class).sum())).item()
        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'correct_per_class_g': test_correct_per_class_p,
                                     'weight_per_class': weight_per_class_dict}

    def set_params(self, model_state_dict, exclude_keys):
        if self.is_on_server == False:
            raise ValueError("Should not be called.")


class FedProtoServer(Server):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.summary_setup()
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        # make sure the starting point is correct
        self.server_side_client.set_on_server()
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.exclude_layer_keys = set(self.server_model_state_dict.keys())
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        if len(self.exclude_layer_keys) > 0:
            print(f"{self.server_config['strategy']}Server: the following keys will not be aggregated:\n ", self.exclude_layer_keys)
        # initial global prototype for server side clients
        dim = self.server_model_state_dict['head.weight'].shape[1]
        self.global_protos = {cls: torch.rand(dim).to(self.server_side_client.device) for cls in range(self.server_config['num_classes'])}
        self.server_side_client.set_protos(self.global_protos)

    def aggregate(self, client_uploads, round):
        uploaded_protos = []
        with torch.no_grad():
            for idx, local_prototypes in enumerate(client_uploads):
                uploaded_protos.append(deepcopy(local_prototypes))
            updated_proptos = proto_aggregation(uploaded_protos)
            for k, v in updated_proptos.items():
                self.global_protos[k] = v

    def testing(self, round, active_only=True, **kwargs):
        """
        active_only: only compute statiscs with to the active clients only
        """
        # get the latest protos
        self.server_side_client.set_protos(self.global_protos)
        # test the performance for global models
        self.server_side_client.testing(round, testloader=None)  # use global testdataset
        print(' server global model correct', torch.sum(self.server_side_client.test_acc_dict[round]['correct_per_class']).item())
        # test the performance for local models (potentiallt only for active local clients)
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        for cid in client_indices:
            client = self.clients_dict[cid]
            # test local model on the splitted testset
            if self.server_config['split_testset'] == True:
                client.testing(round, None)
            else:
                # test local model on the global testset
                client.testing(round, self.server_side_client.testloader)

    def collect_stats(self, stage, round, active_only, **kwargs):
        """
            No actual training and testing is performed. Just collect stats.
            stage: str;
                {"train", "test"}
            active_only: bool;
                True: compute stats on active clients only
                False: compute stats on all clients
        """
        # get client_indices
        client_indices = self.clients_dict.keys()
        if active_only:
            client_indices = self.active_clients_indicies
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        if stage == 'train':
            for cid in client_indices:
                client = self.clients_dict[cid]
                # client.train_loss_dict[round] is a list compose the training loss per end of each epoch
                loss, acc, num_samples = client.train_loss_dict[round][-1], client.train_acc_dict[round][-1], client.num_train_samples
                total_loss += loss * num_samples
                total_acc += acc * num_samples
                total_samples += num_samples
            average_loss, average_acc = total_loss / total_samples, total_acc / total_samples
            self.average_train_loss_dict[round] = average_loss
            self.average_train_acc_dict[round] = average_acc
        else:
            # test stage
            # get global model performance
            self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
            acc_criteria = self.server_side_client.test_acc_dict[round]['acc_by_criteria'].keys()
            # get local model average performance
            self.average_pfl_test_acc_dict[round] = {key: 0.0 for key in acc_criteria}
            for cid in client_indices:
                client = self.clients_dict[cid]
                acc_by_criteria_dict = client.test_acc_dict[round]['acc_by_criteria']
                for key in acc_criteria:
                    self.average_pfl_test_acc_dict[round][key] += acc_by_criteria_dict[key]

            num_participants = len(client_indices)
            for key in acc_criteria:
                self.average_pfl_test_acc_dict[round][key] /= num_participants

    def run(self, **kwargs):
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            setup_seed(r + kwargs['global_seed'])
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
                # dowload prototypes
                client.set_protos(self.global_protos)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
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
            self.aggregate(client_uploads, round=r)
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
