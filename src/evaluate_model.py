import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.data import Subset
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score, accuracy_score
from eoh.problems.ad_examples.models.ML_GCN_model.models import gcn_resnet101_attack
from eoh.problems.ad_examples.models.ml_liw_model.models import inceptionv3_attack
from eoh.problems.ad_examples.models.ml_liw_model.util import *
from eoh.problems.ad_examples.models.ml_liw_model.voc import Voc2012Classification, Voc2007Classification
from eoh.problems.ad_examples.data_nuswide import *
from eoh.problems.ad_examples.data_coco import *

class NewModel(nn.Module):
    def __init__(self, model):
        super(NewModel, self).__init__()
        self.model = model
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, input):
        o = torch.sigmoid(self.model(input))
        return o

def evaluate_model(model, data_loder):
    model.eval()
    output = []
    y = []
    loop = tqdm(enumerate(data_loder), total=len(data_loder))
    with torch.no_grad():
        for i , (imgs, labels) in loop:
            imgs = imgs[0]
            imgs = imgs.cuda()
            o = model(imgs).cpu().numpy()
            output.extend(o)
            y.extend(labels.cpu().numpy())
        output = np.asarray(output)
        y = np.asarray(y)
    pred = (output >= 0.5) + 0
    y[y == -1] = 0
    # correct = np.all((pred == y), axis=1) + 0
    # acc = correct.sum() / len(correct)
    accuray = accuracy_score(y, pred)
    ranking_precison = label_ranking_average_precision_score(y, output)
    print("分类准确率为{:.4f}".format(accuray))
    print("排序精度为{:.4f}".format(ranking_precison))


def main():
    transform = transforms.Compose([Warp(448), transforms.ToTensor()])
    # dataset = Voc2012Classification('./eoh/problems/ad_examples/data', set='mlliw_adv', transform=transform)
    dataset = COCO2014('./eoh/problems/ad_examples/data/coco', phase='mlgcn_adv', transform=transform)
    # dataset = Voc2007Classification('./eoh/problems/ad_examples/data', set='mlliw_adv', transform=transform)
    dataset = Subset(dataset, range(len(dataset)))
    data_loder = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = gcn_resnet101_attack(num_classes=80,
                                 t=0.4,
                                 adj_file='./eoh/problems/ad_examples/data/coco/coco_adj.pkl',
                                 word_vec_file='./eoh/problems/ad_examples/data/coco/coco_glove_word2vec.pkl',
                                 save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlgcn/coco/model_best.pth.tar')
    model = NewModel(model)
    # model = inceptionv3_attack(num_classes=80,
    #                            save_model_path='/home/lyj/LLMs_Multi_Labels/eoh/problems/ad_examples/checkpoint/mlliw/coco/model_best.pth.tar')
    model.cuda()
    evaluate_model(model, data_loder)


if __name__ == '__main__':
    main()