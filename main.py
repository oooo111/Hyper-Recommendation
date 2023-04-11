import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np
from hv import HyperVolume

from models.sharedbottom import SharedBottomModel
from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe import MMoEModel
from models.ple import PLEModel
from models.aitm import AITMModel
from models.metaheac import MetaHeacModel
from models.pomoe import POMoEModel
from sample import circle_points
import pandas as pd
import random

class AliExpressDataset(torch.utils.data.Dataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, dataset_path):
        data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
        self.categorical_data = data[:, :16].astype(np.int)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    
    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'metaheac':
        print("Model: MetaHeac")
        return MetaHeacModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, critic_num=5, dropout=0.2)
    elif name=='pomoe':
        print('Model: pomoe')
        return POMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    #elif name=='psb':
    #    print("Model: Preference-Shared-Bottom")
    #    return PSharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, model_name, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        ray = torch.from_numpy(
            np.random.dirichlet([1, 1.2], 1).astype(np.float32).flatten()
        ).to(device)
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
        if model_name in ['pomoe', 'psb'] :
            y = model(categorical_fields, numerical_fields, ray)
        else:
            y = model(categorical_fields, numerical_fields)

        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        if model_name in ['pomoe', 'psb']:
            for item, weight in zip(loss_list, ray):
                loss += item*weight
            cossim = torch.nn.functional.cosine_similarity(torch.stack(loss_list), 1-ray, dim=0)  ##
            loss -= 0.05*cossim  #(0.04 8个专家,0.05 6个专家)
        else:
            for item in loss_list:
                loss += item
            loss /= len(loss_list)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def metatrain(model, optimizer, data_loader, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
        batch_size = int(categorical_fields.size(0) / 2)
        list_sup_categorical.append(categorical_fields[:batch_size])
        list_qry_categorical.append(categorical_fields[batch_size:])
        list_sup_numerical.append(numerical_fields[:batch_size])
        list_qry_numerical.append(numerical_fields[batch_size:])
        list_sup_y.append(labels[:batch_size])
        list_qry_y.append(labels[batch_size:])
        
        if (i + 1) % 2 == 0:
            loss = model.global_update(list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, task_num, device, model_name, num_rays=5):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    if model_name in ['pomoe', 'psb']:
        for i in range(task_num):
            for j in range(num_rays):
                labels_dict[f"{i}_{j}"], predicts_dict[f"{i}_{j}"], loss_dict[f"{i}_{j}"] = [], [], []
    else:
        for i in range(task_num):
            labels_dict[i], predicts_dict[i], loss_dict[i] = [], [], []
    with torch.no_grad():
        test_rays = circle_points(num_rays, dim=task_num)
        for num, ray in enumerate(test_rays):
            ray = torch.from_numpy(ray.astype(np.float32)).to(device)
            ray /= ray.sum()
            for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
                if model_name in ['pomoe', 'psb']:
                    y = model(categorical_fields, numerical_fields, ray)
                    for i in range(task_num):
                        labels_dict[f"{i}_{num}"].extend(labels[:, i].tolist())
                        predicts_dict[f"{i}_{num}"].extend(y[i].tolist())
                        loss_dict[f"{i}_{num}"].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(),
                                                                                     reduction='none').tolist())
                else:
                    y = model(categorical_fields, numerical_fields)
                    for i in range(task_num):
                        labels_dict[i].extend(labels[:, i].tolist())
                        predicts_dict[i].extend(y[i].tolist())
                        loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
            if model_name not in ['pomoe', 'psb']:
                break
    # print(predicts_dict)
    auc_results, loss_results = list(), list()
    if model_name in ['pomoe', 'psb']:
        for j in range(len(test_rays)):
            auc, losses = [], []
            for i in range(task_num):
                auc.append(roc_auc_score(labels_dict[f"{i}_{j}"], predicts_dict[f"{i}_{j}"]))
                losses.append(np.array(loss_dict[f"{i}_{j}"]).mean())
            auc_results.append(auc)
            loss_results.append(losses)
    else:
        for i in range(task_num):
            auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
            loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results


def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir):
    set_seed(1)
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.csv')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    model = get_model(model_name, field_dims, numerical_num, task_num, expert_num, embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    # criterion = BCEFocalLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_path=f'{save_dir}/{dataset_name}_{model_name}.pt'
    early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    HV = HyperVolume(referencePoint=[1, 1])
    for epoch_i in range(epoch):
        if model_name == 'metaheac':
            metatrain(model, optimizer, train_data_loader, device)
        else:
            train(model, optimizer, train_data_loader, criterion, device, model_name)
        auc, loss = test(model, test_data_loader, task_num, device, model_name)
        volume = HV.compute(1-np.array(auc)) if len(auc) > 2 else (1-(1-auc[0]))*(1-(1-auc[1]))
        print('epoch:{}, AUC {}, Log-loss {}, HV:{}'.format(epoch_i, auc, loss, volume))
        if not early_stopper.is_continuable(model, volume):
            print(f'test: best HV: {early_stopper.best_accuracy}')
            break

    model.load_state_dict(torch.load(save_path))
    auc, loss = test(model, test_data_loader, task_num, device, model_name)
    volume = HV.compute(1-np.array(auc)) if len(auc) > 2 else (1 - (1 - auc[0])) * (1 - (1 - auc[1]))
    f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding = 'utf-8')
    f.write('learning rate: {}\n'.format(learning_rate))
    print('HV {}, AUC {}, Log-loss {}'.format(volume, auc, loss))
    f.write('AUC {}, Log-loss {}, HV {}, Solution {}\n'.format(auc, loss, volume, loss))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL', choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='metaheac', choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac', 'pomoe'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--save_dir', default='./outputs')
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('--lamb', type=int, default=3)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir)
