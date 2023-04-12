import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from training.config import TrainConfig
from prepare_data.create_train_df import create_unique_train_set_df
from training.dataset import VGGDataset
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def tester(config, model_path):
    eps = 0.0001

    test_dataset = VGGDataset(config.df_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #model = get_vgg_model(config.num_classes)
    model=torch.load(model_path, map_location=torch.device('cpu'))
    model.to(device)
    model.eval()
    sensitivity, specificity, acc =[], [], []
    for cos_thresh in [0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9]:
        print (f'testing thresh: {cos_thresh}')
        tp,tn,fp,fn = 0,0,0,0
        cos_loss=0
        for i, data in tqdm(enumerate(test_dataloader, 0)):
            img1 = data['image1'].to(device)
            img2 = data['image2'].to(device)
            label = data['label'].to(device)
            cos=torch.nn.CosineSimilarity()
            outputs1 = model(img1)
            outputs2=model(img2)
            # return_layers={'classifier.0':'classifier'}
            # mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
            # outputs1 = mid_getter(img1.float())[0]['classifier']
            # outputs2 = mid_getter(img2.float())[0]['classifier']
            cos_result = cos(outputs1, outputs2)
            if label==1:
                cos_loss+=(1-cos_result)
            else:
                cos_loss+=cos_result


            pos = cos_result> cos_thresh
            if pos:
                if label == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if label ==0:
                    tn +=1
                else:
                    fn +=1

        print(f'cos loss: {cos_loss / len(test_dataloader)}')

        cur_sens = tp/(tp+fn+eps)
        cur_spec = tn / (tn+fp+eps)
        cur_acc= (tp+tn)/(tp+tn+fp+fn)
        print (cur_sens, cur_spec, cur_acc)
        sensitivity.append(cur_sens)
        specificity.append((1-cur_spec))
        acc.append(cur_acc)
    print (acc)
    plt.plot(specificity, sensitivity)
    plt.xlabel('1-spec')
    plt.ylabel('sens')
    plt.title('ROC model 50 identities, 140 epochs')
    #plt.plot(acc)
    plt.show()



if __name__ == '__main__':
    model_path = r"C:\Users\shiri\Downloads\450_classes_0.01_lr_batch_128_pretrained_normalized_epoch_120.pt"
    config = TrainConfig()

    tester(config, model_path)