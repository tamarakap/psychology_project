import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from training.config import TrainConfig
from prepare_data.create_train_df import create_unique_train_set_df
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from training.dataset import get_datasets
from training.dataset import TestDataset
from training.model import load_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_vgg(config, df_path, weights_dir):
    epochs = config.epochs
    writer = SummaryWriter(comment = config.exp_name)

    train_dataset, val_dataset = get_datasets(df_path, config)
    test_dataset = TestDataset(config.df_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = load_model(config.num_classes, config.model_type)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)
    acc=torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes).to(device)
    best_loss = 100

    # Train loop
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_train_loss, epoch_train_acc = [], []

        for i, data in enumerate(train_dataloader):
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if config.model_type == 'resnet':
                model.classify = True
                outputs = model.forward(inputs.float())
            else:
                outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            thresh_probabilities = torch.argmax(outputs, dim= 1)
            cur_train_acc = acc(thresh_probabilities, labels)


            # print statistics
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(cur_train_acc.to(device))
            if i % 10 ==0: # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(epoch_train_loss):.3f}, acc: {np.mean(epoch_train_acc)}')
        scheduler.step()
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


        # Validation loop
        model.eval()
        epoch_val_loss, epoch_val_acc = [], []
        for i, data in enumerate(val_dataloader):
            inputs = data['image'].to(device)
            labels = data['img_class'].to(device)

            # forward
            if config.model_type == 'resnet':
                outputs = model.forward(inputs.float())
            else:
                outputs = model(inputs.float())

            loss = criterion(outputs, labels)
            thresh_probabilities = torch.argmax(outputs, dim=1)
            cur_val_acc = acc(thresh_probabilities, labels)
            epoch_val_loss.append(loss.item())
            epoch_val_acc.append(cur_val_acc.to(device))

            if loss<best_loss:
                torch.save(model, f'{weights_dir}/epoch_{epoch}_loss_{loss}')
                best_loss=loss

        print(f'[epoch: {epoch + 1}/{epochs}] loss: {np.mean(epoch_val_loss):.3f}, acc: {np.mean(epoch_val_acc)}')
        writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
        writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)


        # Testing on test df
        if config.run_tester:
            epoch_test_mean_cos, epoch_test_thresh_cos = [], []
            for i, data in enumerate(test_dataloader, 0):
                img1 = data['image1'].to(device)
                img2 = data['image2'].to(device)
                label = data['label'].to(device)
                cos=torch.nn.CosineSimilarity()

                return_layers={'classifier.3':'classifier'}
                mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
                outputs1 = mid_getter(img1.float())[0]['classifier']
                outputs2 = mid_getter(img2.float())[0]['classifier']
                cos_result = cos(outputs1, outputs2)
                pos = cos_result> config.cos_thresh


                epoch_test_mean_cos.append(cos_result.detach().numpy())
                epoch_test_thresh_cos.append((pos==label).detach().numpy())

            print(f'[Tester: {epoch + 1}/{epochs}] loss: {np.mean(epoch_test_mean_cos):.3f}, acc: {np.mean(epoch_test_thresh_cos)}')

    print('Finished Training')


if __name__ == '__main__':
    config = TrainConfig()
    num_classes = config.num_classes
    num_instances_per_class = config.train_num_instances_per_class
    exp_dir = f'{config.results_dir}/exps/{config.exp_name}_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'
    weights_dir = f'{exp_dir}/weights'
    df_dir = f'{config.root_dir}/training_set_dfs'
    for dir in [exp_dir, weights_dir, df_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    # df that contains the training and validation data and paths, unique per combination of #classes/#num_instances
    df_path = f'{df_dir}/{num_classes}_classes_{num_instances_per_class}_per_class.csv'
    # if df already exists, use it, else create new one and save
    if os.path.isfile(df_path) and config.make_new_train_set == False:
        df = pd.read_csv(df_path)
    else:
        df = create_unique_train_set_df(config)
        df.to_csv(df_path)
    # In both cases, also save the df to the exp dir, for reproducibility
    df.to_csv(f'{exp_dir}/training_set_{num_classes}_classes_{num_instances_per_class}_per_class.csv')

    train_vgg(config, df_path, weights_dir)