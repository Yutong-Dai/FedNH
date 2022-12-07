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
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
from torch.utils.data.dataloader import DataLoader, Dataset


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def match_loss(gw_syn, gw_real, device, dis_metric):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        # return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


class CReFFClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        # num_classes = self.client_config['num_classes']
        # d = []
        # for k in range(num_classes):
        #     d.append(self.count_by_class[k])
        # print(f"Cid:{cid}", d)

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        self.retrained_model = eval(self.client_config['model'])(self.client_config).to(self.device)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
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
        # print('lr:', optimizer.param_groups[0]['lr'])
        for i in range(num_epochs):
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

    def upload(self):
        return self.new_state_dict

    def testing(self, round, testloader=None):
        self.model.eval()
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        num_classes = self.client_config['num_classes']
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in range(num_classes)])
        test_correct_per_class = torch.tensor([0] * num_classes)

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
                yhat = self.model.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}

    def compute_real_feature_gradient(self, global_retrained_model_state_dict):
        list_class, per_class_compose = list(self.count_by_class.keys()), list(self.count_by_class.values())
        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}
        images_all = [torch.unsqueeze(self.trainset[i][0], dim=0) for i in range(len(self.trainset))]
        try:
            labels_all = [self.trainset[i][1].item() for i in range(len(self.trainset))]
        except AttributeError:
            labels_all = [self.trainset[i][1] for i in range(len(self.trainset))]

        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        # labels_all = torch.tensor(labels_all, dtype=torch.long)

        def get_images(c, n, indices_class):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        self.retrained_model.set_params(global_retrained_model_state_dict, exclude_keys=set())
        self.retrained_model.eval()
        self.retrained_model.prototype.train()
        net_parameters = list(self.retrained_model.prototype.parameters())
        criterion = nn.CrossEntropyLoss().to(self.device)
        # gradients of all classes
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}
        # choose to repeat 10 times
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, self.client_config['CReFF_batch_real'], indices_class).to(self.device)
                lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                feature_real, output_real = self.retrained_model.get_embedding(img_real)

                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg


class CReFFServer(Server):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, **kwargs)
        self.summary_setup()
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        # make sure the starting point is correct
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.device = self.server_side_client.device
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        if len(self.exclude_layer_keys) > 0:
            print(f"{self.server_config['strategy']}Server: the following keys will not be aggregated:\n ", self.exclude_layer_keys)

        # additional params for CReFF
        prototype = self.server_side_client.model.prototype
        self.lantent_dim, self.num_classes = prototype.in_features, prototype.out_features
        self.num_of_feature = self.server_config['CReFF_num_of_fl_feature']
        self.feature_syn = torch.randn(size=(self.num_classes * self.num_of_feature,
                                             self.lantent_dim), dtype=torch.float,
                                       requires_grad=True, device=self.device)
        self.label_syn = torch.tensor([np.ones(self.num_of_feature) * i for i in range(self.num_classes)],
                                      dtype=torch.long,
                                      requires_grad=False, device=self.device).view(-1)
        self.optimizer_feature = torch.optim.SGD([self.feature_syn, ], lr=self.server_config['CReFF_lr_feature'])  # optimizer_img for synthetic data
        self.feature_criterion = nn.CrossEntropyLoss().to(self.device)
        self.feature_net = nn.Linear(self.lantent_dim, self.num_classes, bias=False).to(self.device)
        self.batch_size = self.clients_dict[0].client_config['batch_size']

    def update_feature_syn(self, global_params, list_clients_gradient):
        """
        list_clients_gradient: a list of dict, 
        each dict contains gradient (value) per class (ket) from 

        only use the fc parameter from global_params
        """
        feature_net_params = self.feature_net.state_dict()
        # no bias term in the linear layer
        feature_net_params['weight'] = global_params['prototype.weight']
        self.feature_net.load_state_dict(feature_net_params)
        self.feature_net.train()
        net_global_parameters = list(self.feature_net.parameters())
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(self.num_classes)}
        # aggregate the real feature gradients
        for i in range(self.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]
            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                # only has the gradient for weight not bias
                for name_param in range(1):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        for ep in range(self.server_config['CReFF_match_epoch']):
            loss_feature = torch.tensor(0.0).to(self.device)
            for c in range(self.num_classes):
                if len(gw_real_avg[c]) != 0:
                    feature_syn = self.feature_syn[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, self.lantent_dim))
                    lab_syn = torch.ones((self.num_of_feature,), device=self.device, dtype=torch.long) * c
                    output_syn = self.feature_net(feature_syn)
                    loss_syn = self.feature_criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], self.device, self.server_config['CReFF_dis_metric'])
            self.optimizer_feature.zero_grad()
            loss_feature.backward()
            self.optimizer_feature.step()

    def feature_re_train(self, fedavg_params, batch_size_local_training):
        feature_syn_train_ft = deepcopy(self.feature_syn.detach())
        label_syn_train_ft = deepcopy(self.label_syn.detach())
        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(self.lantent_dim, self.num_classes).to(self.device)
        optimizer_ft_net = torch.optim.SGD(ft_model.parameters(), lr=self.server_config['CReFF_lr_net'])  # optimizer_img for synthetic data
        ft_model.train()
        trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                    batch_size=batch_size_local_training,
                                    shuffle=True)
        for epoch in range(self.server_config['CReFF_crt_epoch']):
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.feature_criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        feature_net_params = ft_model.state_dict()
        fedavg_params['prototype.weight'] = feature_net_params['weight']
        return deepcopy(ft_model.state_dict()), deepcopy(fedavg_params)
        # re_trained_full_model_params = deepcopy(fedavg_params)
        # re_trained_full_model_params['prototype.weight'] = ft_model.state_dict()['weight']
        # return deepcopy(ft_model.state_dict()), deepcopy(re_trained_full_model_params)

    def aggregate(self, client_uploads, round):
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        exclude_layer_keys = self.exclude_layer_keys
        with torch.no_grad():
            for idx, client_state_dict in enumerate(client_uploads):
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              exclude=exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=exclude_layer_keys
                                                                                )
            # new global model
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         server_lr / num_participants,
                                                                         exclude=exclude_layer_keys
                                                                         )

    def testing(self, round, active_only=True, **kwargs):
        """
        active_only: only compute statiscs with to the active clients only
        """
        #  use retrained model
        self.server_side_client.set_params(self.retrained_full_model_params, self.exclude_layer_keys)
        # test the performance for global models
        self.server_side_client.testing(round, testloader=None)  # use global testdataset
        # revert back server_model_state_dict is fedavg updatde weight
        self.server_side_client.set_params(self.server_model_state_dict, self.exclude_layer_keys)
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
        syn_params = deepcopy(self.feature_net.state_dict())
        # round index begin with 1
        for r in round_iterator:
            syn_feature_params = deepcopy(self.server_model_state_dict)
            syn_feature_params['prototype.weight'] = syn_params['weight']

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
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            list_clients_gradient = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                # compute the real feature gradients in local data
                truth_gradient = client.compute_real_feature_gradient(deepcopy(syn_feature_params))
                list_clients_gradient.append(deepcopy(truth_gradient))
                # local update
                # trian as fedavg model params are ready set.
                client.training(r, client.client_config['num_epochs'])
                client_uploads.append(client.upload())
            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            self.aggregate(client_uploads, round=r)

            self.update_feature_syn(deepcopy(syn_feature_params), list_clients_gradient)
            # re-trained classifier
            # syn_params only contains head
            syn_params, self.retrained_full_model_params = self.feature_re_train(deepcopy(self.server_model_state_dict), self.batch_size)

            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")
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

    def save(self, filename, keep_clients_model=False):
        if not keep_clients_model:
            for client in self.clients_dict.values():
                client.model = None
                client.retrained_model = None
                client.trainloader = None
                client.trainset = None
                client.new_state_dict = None
        self.server_side_client.trainloader = None
        self.server_side_client.trainset = None
        self.server_side_client.testloader = None
        self.server_side_client.testset = None
        save_to_pkl(self, filename)
