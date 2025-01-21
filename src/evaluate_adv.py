import os
import logging

from sympy.physics.units import length
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from eoh.problems.ad_examples.models.ml_liw_model.voc import Voc2012Classification, Voc2007Classification
from eoh.problems.ad_examples.models.ml_liw_model.util import *
from eoh.problems.ad_examples.models.ML_GCN_model.models import gcn_resnet101_attack
from eoh.problems.ad_examples.models.ml_liw_model.models import inceptionv3_attack
from eoh.problems.ad_examples.data_nuswide import NusWide
from eoh.problems.ad_examples.data_coco import COCO2014


class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        o = torch.sigmoid(self.model(input))
        return o

def get_model(paras):
    if paras.model == 'mlgcn':
        model = gcn_resnet101_attack(num_classes=paras.num_classes,
                                     t=0.4,
                                     adj_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                              + paras.dataset + '/' + paras.dataset + '_adj.pkl',
                                     word_vec_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                                   + paras.dataset + '/glove_word2vec.pkl',
                                     save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlgcn/'
                                                     + paras.dataset + '/model_best.pth.tar')
        model = NewModel(model)
    elif paras.model == 'mlliw':
        model = inceptionv3_attack(num_classes=paras.num_classes,
                                   save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlliw/'
                                                   + paras.dataset + '/model_best.pth.tar')
    else:
        print("error model")
        exit(0)
    return model


def get_data(paras):
    if paras.dataset == "voc2012":
        data = Voc2012Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data',
                                     set=paras.model + '_adv')
        data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
        data = Subset(data, range(len(data)))
        data = DataLoader(data, batch_size=32, shuffle=False, num_workers=4)
    elif paras.dataset == "voc2007":
        data = Voc2007Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data',
                                     set=paras.model + '_adv')
        data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
        data = Subset(data, range(len(data)))
        data = DataLoader(data, batch_size=32, shuffle=False, num_workers=4)
    elif paras.dataset == "nuswide":
        data = NusWide(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/nuswide',
                       phase=paras.model + '_adv')
        data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
        data = Subset(data, range(len(data)))
        data = DataLoader(data, batch_size=32, shuffle=False, num_workers=4)
    elif paras.dataset == "coco":
        data = COCO2014(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/coco',
                        phase=paras.model + '_adv')
        data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
        data = Subset(data, range(len(data)))
        data = DataLoader(data, batch_size=32, shuffle=False, num_workers=4)
    else:
        print("error dataset")
        exit(0)
    return data


def evaluate_adv(paras, attack_method):
    model = get_model(paras)
    model = model.cuda()
    model.eval()
    adv_folder_path = '../adv_imgs/' + paras.model + '/' + paras.dataset + '/' + attack_method + '/'
    org_folder_path = '../org_imgs/' + paras.model + '/' + paras.dataset + '/'
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith('show_all')]
    org_file_list = [file for file in os.listdir(org_folder_path)]
    adv_file_list.sort(key=lambda x: int(x[9: -4]))
    org_file_list.sort(key=lambda x: int(x[8: -4]))
    adv = []
    org = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    for i, f in enumerate(org_file_list):
        org.extend(np.load(org_folder_path + f))

    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv, batch_size=32, shuffle=False, num_workers=4)
    dl2 = torch.utils.data.DataLoader(org, batch_size=32, shuffle=False, num_workers=4)
    dl2 = tqdm(dl2, desc='ADV')
    adv_output = []
    norm = []
    norm_1 = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x.cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x) * 255) - ((batch_test_x) * 255)
            batch_norm = [np.linalg.norm(r) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))

        adv_output = np.asarray(adv_output)
        adv_pred = adv_output.copy()
        adv_pred[adv_pred >= (0.5 + 0)] = 1
        adv_pred[adv_pred < (0.5 + 0)] = -1

        norm_t = np.asarray(norm)
        max_r_t = np.asarray(max_r)
        mean_r_t = np.asarray(mean_r)
        rmsd_t = np.asarray(rmsd)
        norm_1_t = np.asarray(norm_1)

        adv_pred_match_target = np.all((adv_pred == paras.target.cpu().numpy()), axis=1) + 0
        attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
        norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
        max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
        norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
        mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
        rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)
        metrics = dict()
        adv_pred[adv_pred == -1] = 0
        metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
        metrics['norm'] = np.mean(norm_t)
        metrics['norm_1'] = np.mean(norm_1_t)
        metrics['rmsd'] = np.mean(rmsd_t)
        metrics['max_r'] = np.mean(max_r_t)
        metrics['mean_r'] = np.mean(mean_r_t)
        metrics['mean_pos_label'] = np.sum(adv_pred) / len(adv_pred)
        print()
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info(str(metrics))


def evaluate_many_adv(paras, attack_method):
    model = get_model(paras)
    model = model.cuda()
    model.eval()
    adv_folder_path = '../adv_imgs/' + paras.model + '/' + paras.dataset + '/' + attack_method + '/'
    org_folder_path = '../org_imgs/' + paras.model + '/' + paras.dataset + '/'
    adv_file_list = [file for file in os.listdir(adv_folder_path) if file.startswith('show_all')]
    org_file_list = [file for file in os.listdir(org_folder_path)]
    adv_file_list.sort(key=lambda x: int(x[9: -4]))
    org_file_list.sort(key=lambda x: int(x[8: -4]))
    adv = []
    org = []

    for i, f in enumerate(adv_file_list):
        adv.extend(np.load(adv_folder_path + f))

    for i, f in enumerate(org_file_list):
        org.extend(np.load(org_folder_path + f))

    adv = np.asarray(adv)
    org = np.asarray(org)
    data_len = min(len(adv), len(org))
    adv, org = adv[:data_len], org[:data_len]
    dl1 = torch.utils.data.DataLoader(adv, batch_size=32, shuffle=False, num_workers=4)
    dl2 = torch.utils.data.DataLoader(org, batch_size=32, shuffle=False, num_workers=4)
    dl2 = tqdm(dl2, desc='ADV')
    adv_output = []
    norm = []
    norm_1 = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x.cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x) * 255) - ((batch_test_x) * 255)
            batch_norm = [np.linalg.norm(r) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))

        adv_output = np.asarray(adv_output)
        adv_pred = adv_output.copy()
        adv_pred[adv_pred >= (0.5 + 0)] = 1
        adv_pred[adv_pred < (0.5 + 0)] = 0

        if paras.dataset == 'voc2012' or paras.dataset == 'voc2007':
            add_of_means = [20 * rate for rate in [0.25, 0.5, 0.75, 1]]
        elif paras.dataset == 'coco':
            add_of_means = [80 * rate for rate in [0.25, 0.5, 0.75, 1]]
        elif paras.dataset == 'nuswide':
            add_of_means = [81 * rate for rate in [0.25, 0.5, 0.75, 1]]

        for i, add_of_mean in enumerate(add_of_means):
            norm_t = np.asarray(norm)
            max_r_t = np.asarray(max_r)
            mean_r_t = np.asarray(mean_r)
            rmsd_t = np.asarray(rmsd)
            norm_1_t = np.asarray(norm_1)

            adv_pred_match_target = (np.sum(adv_pred, axis=1) >= int(add_of_mean)) + 0
            attack_fail_idx = np.argwhere(adv_pred_match_target == 0).flatten().tolist()
            norm_t = np.delete(norm_t, attack_fail_idx, axis=0)
            max_r_t = np.delete(max_r_t, attack_fail_idx, axis=0)
            norm_1_t = np.delete(norm_1_t, attack_fail_idx, axis=0)
            mean_r_t = np.delete(mean_r_t, attack_fail_idx, axis=0)
            rmsd_t = np.delete(rmsd_t, attack_fail_idx, axis=0)
            metrics = dict()
            adv_pred[adv_pred == -1] = 0
            metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
            metrics['norm'] = np.mean(norm_t)
            metrics['norm_1'] = np.mean(norm_1_t)
            metrics['rmsd'] = np.mean(rmsd_t)
            metrics['max_r'] = np.mean(max_r_t)
            metrics['mean_r'] = np.mean(mean_r_t)
            metrics['mean_pos_label'] = np.sum(adv_pred) / len(adv_pred)
            print()
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info(str(metrics))


