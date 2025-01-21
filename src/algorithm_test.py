import torch
from torch.autograd import Variable
from tqdm import tqdm
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

class GENADEXAMPLES:
    def __init__(self, target, model_name, data_name):
        self.target = target
        self.model_name = model_name
        self.data_name = data_name
        self.maxiter = 30
        self.epsilon = 0.3
        self.alpha = 0.02
        self.loss_func = nn.BCELoss(reduction='mean')
        self.model = self.get_model(self.model_name)
        self.data = self.get_dataset(self.data_name)

    def generate(self):
        torch.cuda.set_device(0)
        self.model = self.model.cuda()
        self.model.eval()
        loop = tqdm(enumerate(self.data), total=len(self.data))
        l2_norm = []
        adv_outputs = []
        correct = []
        pos_num = []
        for idx, (data, label) in loop:
            data = data[0]
            label[label == -1] = 0
            data, label = data.cuda(), label.cuda()
            init_pre = self.model(data)
            init_pre[init_pre >= 0.5] = 1
            init_pre[init_pre < 0.5] = 0
            index_correct = torch.where(torch.all((init_pre == label), axis=1))
            data = data[index_correct]
            label = label[index_correct]

            # data = data.cpu().detach().numpy()
            # save_adv_path = ('./org_imgs/' + self.model_name + '/' +
            #                       self.data_name + '/' + 'org_imgs' + str(idx) + '.npy')
            # np.save(save_adv_path, data)
            target = self.target.repeat(len(data), 1).cuda()
            adv_img = self.gen_adv_examples(data, self.model, target)
            adv_lab = self.model(adv_img)
            adv_lab[adv_lab >= 0.5] = 1
            adv_lab[adv_lab < 0.5] = 0
            correct.extend(torch.all((adv_lab == target), axis=1).cpu().numpy())
            pos_num.extend(adv_lab.eq(target).view(-1).cpu().numpy())
            adv_img = adv_img.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            batch_r = adv_img - data
            batch_l2_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            l2_norm.extend(batch_l2_norm)
            adv_outputs.extend(adv_img)
            save_adv_path = ('../adv_imgs/' + self.model_name + '/' +
                             self.data_name + '/' + 'gpt/' + 'show_all_' + str(idx) + '.npy')
            np.save(save_adv_path, adv_img)
        accuracy = sum(correct) / len(correct)
        neg_num = pos_num.count(False)
        l2_norm = np.asarray(l2_norm)
        l2_norm = np.mean(l2_norm)
        print("Accuracy: {:.4f}".format(accuracy))
        print("L2 norm: {:.4f}".format(l2_norm))
        print("平均正标签数: {:.4f}".format(pos_num.count(True) / len(correct)))

    def i_fgsm(self, org_img, model, target):
        adv_img = Variable(org_img.data, requires_grad=True)
        for _ in range(self.maxiter):
            grad = self.get_gradient(adv_img, model, target)
            adv_img = adv_img + self.alpha * grad.sign()
            adv_img = self.clip_adv(org_img, adv_img)
        return adv_img

    def mi_fgsm(self, org_img, model, target):
        g = 0
        decay_factor = 1.0
        adv_img = Variable(org_img.data, requires_grad=True)
        for _ in range(self.maxiter):
            grad = self.get_gradient(adv_img, model, target)
            g = decay_factor * g + grad / torch.norm(adv_img.grad, p=1, dim=[1, 2, 3], keepdim=True)
            adv_img = adv_img + self.alpha * g.sign()
            adv_img = self.clip_adv(org_img, adv_img)
        return adv_img

    def PGD(self, org_img, model, target):
        adv_img = org_img + torch.empty_like(org_img).uniform_(-self.epsilon, self.epsilon)
        adv_img = Variable(adv_img.data, requires_grad=True)
        for _ in range(self.maxiter):
            grad = self.get_gradient(adv_img, model, target)
            adv_img = adv_img + self.alpha * adv_img.grad.sign()
            adv_img = self.clip_adv(org_img, adv_img)
        return adv_img

    def gen_adv_examples(self, org_img, model, target):
        """
        {This algorithm introduces a cosine annealing learning rate to adjust the step size dynamically and uses a different decay factor to enhance the generation of adversarial examples while maintaining visual similarity to the original images.}
        """
        adv_img = Variable(org_img.data, requires_grad=True)
        momentum = torch.zeros_like(org_img)
        decay_factor = 0.9  # Different decay factor
        eta_max = 0.2  # Maximum value for cosine annealing adjustment
        eta_min = 0.05  # Minimum value for cosine annealing adjustment

        for i in range(self.maxiter):
            grad = self.get_gradient(adv_img, model, target)
            momentum = decay_factor * momentum + grad / torch.norm(grad, p=1, dim=[1, 2, 3], keepdim=True)

            # Cosine annealing for dynamic step size
            eta_t = eta_min + 0.5 * (eta_max - eta_min) * (
                    1 + torch.cos(torch.tensor(i / self.maxiter * 3.141592653589793)))
            step = eta_t * self.alpha * momentum.sign()

            adv_img = adv_img + step
            adv_img = self.clip_adv(org_img, adv_img)

        return adv_img

    def get_model(self, model_name):
        if model_name == 'mlgcn':
            model = gcn_resnet101_attack(num_classes=num_class,
                                         t=0.4,
                                         adj_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                                  + self.data_name + '/' + self.data_name + '_adj.pkl',
                                         word_vec_file='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/'
                                                       + self.data_name + '/glove_word2vec.pkl',
                                         save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlgcn/'
                                                         + self.data_name + '/model_best.pth.tar')
            model = NewModel(model)
        elif model_name == 'mlliw':
            model = inceptionv3_attack(num_classes=num_class,
                                       save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlliw/'
                                                       + self.data_name + '/model_best.pth.tar')
        else:
            print("error model")
            exit(0)
        return model

    def get_dataset(self, data_name):
        if data_name == "voc2012":
            data = Voc2012Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data', set=self.model_name + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(len(data)))
            data = DataLoader(data, batch_size=10, shuffle=False)
        elif data_name == "voc2007":
            data = Voc2007Classification(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data', set=self.model_name + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(len(data)))
            data = DataLoader(data, batch_size=10, shuffle=False)
        elif data_name == "nuswide":
            data = NusWide(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/nuswide', phase=self.model_name + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(len(data)))
            data = DataLoader(data, batch_size=10, shuffle=False)
        elif data_name == "coco":
            data = COCO2014(root='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/data/coco', phase=self.model_name + '_adv')
            data.transform = transforms.Compose([Warp(448), transforms.ToTensor()])
            data = Subset(data, range(len(data)))
            data = DataLoader(data, batch_size=10, shuffle=False)
        else:
            print("error dataset")
            exit(0)
        return data

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

num_class = 80
target = torch.tensor([1] * num_class)
a = GENADEXAMPLES(target, 'mlliw', 'coco')
a.generate()

