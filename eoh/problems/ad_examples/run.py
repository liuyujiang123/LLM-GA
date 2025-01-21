import torch
import warnings
import types
import sys

from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from .prompts import GetPrompts
from .models.ml_liw_model.voc import Voc2007Classification, Voc2012Classification
from .models.ml_liw_model.util import *
from .models.ML_GCN_model.models import gcn_resnet101_attack
from .models.ml_liw_model.models import inceptionv3_attack
from .data_nuswide import *
from .data_coco import *
from eoh.methods.eoh import global_val as gv


class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        o = torch.sigmoid(self.model(input))
        return o

class GENADEXAMPLES:
    def __init__(self, paras):
        self.paras = paras
        self.target = paras.target
        self.maxiter = 30
        self.epsilon = 0.3
        self.alpha = 0.02
        self.loss_func = nn.BCELoss(reduction='mean')
        self.prompts=GetPrompts()
        self.num_class = self.get_num_class(paras.dataset)
        self.model = self.get_model(paras.model)
        self.data = self.get_dataset(paras.dataset)

    def generate(self, eva, device_id):
        # if device_id == 7:
        #     device_id = 3
        # if device_id == 8:
        #     device_id = 4
        # if device_id == 9:
        #     device_id = 5
        # if device_id == 10:
        #     device_id = 6
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        print(f"当前运行的是gpu: {device_id}")
        self.model = self.model.to(device)
        self.model.eval()
        loop = tqdm(enumerate(self.data), total=len(self.data))
        l2_norm = []
        adv_outputs = []
        correct = []
        pos_num = []
        for idx, (data, label) in loop:
            data = data[0]
            label[label == -1] = 0
            data, label = data.to(device), label.to(device)
            init_pre = self.model(data)
            init_pre[init_pre >= 0.5] = 1
            init_pre[init_pre < 0.5] = 0
            index_correct = torch.where(torch.all((init_pre == label), axis=1))
            data = data[index_correct]
            label = label[index_correct]
            target = self.target.repeat(len(data), 1).to(device)
            adv_data = eva.gen_adv_examples(self, data, self.model, target)
            adv_lab = self.model(adv_data)
            adv_lab[adv_lab >= 0.5] = 1
            adv_lab[adv_lab < 0.5] = 0
            correct.extend(torch.all((adv_lab == target), axis=1).cpu().numpy())
            pos_num.extend(adv_lab.eq(target).view(-1).cpu().numpy())
            adv_data = adv_data.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            batch_r = adv_data - data
            batch_l2_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            l2_norm.extend(batch_l2_norm)
            adv_outputs.extend(adv_data)
        accuracy = sum(correct) / len(correct)
        neg_num = pos_num.count(False)
        l2_norm = np.asarray(l2_norm)
        l2_norm = np.mean(l2_norm)
        fitness = neg_num - (accuracy * 10) + (l2_norm / 2)
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return float(fitness)

    def evaluate(self, code_string, device_id):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Now you can use the module as you would any other
                fitness = self.generate(heuristic_module, device_id)
                return fitness
        except Exception as e:
            print("Error:", str(e))
            return None

    def get_model(self, model_name):
        if model_name == 'mlgcn':
            model = gcn_resnet101_attack(num_classes=self.num_class,
                                         t=0.4,
                                         adj_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                                  + self.paras.dataset + '/' + self.paras.dataset + '_adj.pkl',
                                         word_vec_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                                       + self.paras.dataset + '/glove_word2vec.pkl',
                                         save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlgcn/'
                                                         + self.paras.dataset + '/model_best.pth.tar')
            model = NewModel(model)
        elif model_name == 'mlliw':
            model = inceptionv3_attack(num_classes=self.paras.num_classes,
                                       save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlliw/' +
                                                       self.paras.dataset + '/model_best.pth.tar')
        else:
            print("error model")
            exit(0)
        return model

    def get_dataset(self, data_name):
        if data_name == "voc2012":
            data = Voc2012Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data', set=self.paras.model + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(50))
            data = DataLoader(data, batch_size=1, shuffle=False)
        elif data_name == "voc2007":
            data = Voc2007Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data', set=self.paras.model + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(50))
            data = DataLoader(data, batch_size=1, shuffle=False)
        elif data_name == "nuswide":
            data = NusWide(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/nuswide', phase=self.paras.model + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(50))
            data = DataLoader(data, batch_size=1, shuffle=False)
        elif data_name == "coco":
            data = COCO2014(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/coco', phase=self.paras.model + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(50))
            data = DataLoader(data, batch_size=1, shuffle=False)
        else:
            print("error dataset")
            exit(0)
        return data

    def get_num_class(self, data_name):
        if data_name == "voc2012":
            return 20
        elif data_name == "voc2007":
            return 20
        elif data_name == "nuswide":
            return 81
        elif data_name == "coco":
            return 80
        else:
            print("error dataset")
            exit(0)

    def get_gradient(self, x, model, target):
        output = model(x)
        true_label = output.clone()
        true_label[true_label >= 0.5] = 1
        true_label[true_label < 0.5] = 0
        loss = -self.loss_func(output, target.float())
        model.zero_grad()
        if x.grad is not None:
            x.grad.data.zero_()
        loss.backward()
        return x.grad.data

    def clip_adv(self, org_img, adv_img):
        adv_img = torch.clamp(adv_img, org_img - self.epsilon, org_img + self.epsilon)
        adv_img = torch.clamp(adv_img, 0, 1)
        adv_img = Variable(adv_img.data, requires_grad=True)
        return adv_img
