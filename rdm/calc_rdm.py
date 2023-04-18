import os

import PIL.JpegImagePlugin
import numpy as np
import torchvision.transforms as det_transforms
from tqdm import tqdm
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from argparse import ArgumentParser
from openpyxl import load_workbook

# torch.manual_seed(42)
from training.dataset_utils import Rescale

cos = torch.nn.CosineSimilarity()
mtcnn = MTCNN(image_size=160)
from facenet_pytorch import InceptionResnetV1

normalize = det_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
composed_transforms = det_transforms.Compose([Rescale((160, 160)), normalize])

vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)


# resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

def get_resnet_model(pretrain='vggface2'):
    # options are: vggface2, casia-webface https://github.com/timesler/facenet-pytorch
    model = InceptionResnetV1(pretrained=pretrain).eval()
    return model


"""
def get_vgg_model(num_classes):
    model = vgg_model
    #model.fc = nn.Linear(in_features = 2048, out_features=num_classes+1, bias = True)
    model.classifier[6] = nn.Linear(in_features = 4096, out_features=num_classes+1, bias = True)
    return model
"""


def load_data(data_dir, instance_num_per_class):
    data_paths_list = []
    for c in os.listdir(data_dir):
        class_path = os.path.join(data_dir, c)
        imgs = os.listdir(class_path)
        for img in imgs[:2]:
            im_path = os.path.join(class_path, img)
            data_paths_list.append(im_path)

    return data_paths_list


def calculate_thacher_index(dif_up, dif_inv):
    return (dif_up - dif_inv) / (dif_up + dif_inv)


def get_image_embedding(img_path, model):
    img = Image.open(img_path)
    img.thumbnail((432, 578))
    img.save(img_path)
    img.show()
    convert_tensor = det_transforms.ToTensor()
    img_as_tensor = convert_tensor(img)
    return model(img_as_tensor.unsqueeze(0).float())


def load_data_for_thacher(data_dir):
    data_paths_list = []
    data_paths_list2 = []
    imgs = os.listdir(data_dir)

    for img in imgs[:2]:
        im_path = os.path.join(data_dir, img)
        data_paths_list.append(im_path)

    for img in imgs[2:]:
        im_path = os.path.join(data_dir, img)
        data_paths_list2.append(im_path)

    return data_paths_list, data_paths_list2


def load_data_for_context_test(data_dir):
    return context_test


def get_context_test_results(dataset_paths_list, model):
    rdm = np.zeros(3)
    imgs = os.listdir(dataset_paths_list)
    # try:
    img_path1 = os.path.join(dataset_paths_list, imgs[0])
    img_path2 = os.path.join(dataset_paths_list, imgs[1])
    img_path3 = os.path.join(dataset_paths_list, imgs[2])

    img_embedding1 = get_image_embedding(img_path1, model)
    img_embedding2 = get_image_embedding(img_path2, model)
    img_embedding3 = get_image_embedding(img_path3, model)

    rdm[0] = cos(img_embedding1, img_embedding2)
    rdm[1] = cos(img_embedding2, img_embedding3)
    rdm[2] = cos(img_embedding1, img_embedding3)
    # except:
    print(f'error calculating similarity')
    return rdm


def get_rdm_thacher(dataset_paths_list_up, dataset_paths_list_inv, model):
    rdm = np.zeros(3)
    try:
        im_path = dataset_paths_list_up[0]
        img = Image.open(im_path)
        img = mtcnn(img)

        img_embedding = model(img.unsqueeze(0).float())
        # a = img.permute(1,2,0).numpy()
        # save_image(a, 'C:/Users/tamarak/Desktop/tamara_university/psycology_seminar/results/thatcherResults/images/0')

        second_im_path = dataset_paths_list_up[1]
        second_img = Image.open(second_im_path)
        second_img = mtcnn(second_img)
        second_img_embedding = model(second_img.unsqueeze(0).float())

        rdm[0] = cos(img_embedding, second_img_embedding)
    except:
        print(f'error calculating similarity for: {im_path}')

    try:
        im_path = dataset_paths_list_inv[0]
        img = Image.open(im_path)
        img = mtcnn(img)
        img = torch.rot90(img, 2)

        img_embedding = model(img.unsqueeze(0).float())
        # a = img.permute(1,2,0).numpy()
        # save_image(a, 'C:/Users/tamarak/Desktop/tamara_university/psycology_seminar/results/thatcherResults/images/1')

        second_im_path = dataset_paths_list_inv[1]
        second_img = Image.open(second_im_path)
        second_img = mtcnn(second_img)
        second_img = torch.rot90(second_img, 2)
        second_img_embedding = model(second_img.unsqueeze(0).float())

        rdm[1] = cos(img_embedding, second_img_embedding)
    except:
        print(f'error calculating similarity for: {im_path}')
    rdm[2] = calculate_thacher_index(rdm[0], rdm[1])
    return rdm


# C:\Users\tamarak\Desktop\tamara_university\psycology_seminar\img_by_identity\img24_plot\img24
# get repr. from model
def get_rdm(dataset_paths_list, model):
    rdm = np.zeros((len(dataset_paths_list), len(dataset_paths_list)))
    for i, im_path in tqdm(enumerate(dataset_paths_list)):
        try:
            img = Image.open(im_path)
            img = mtcnn(img)
            img_embedding = model(img.unsqueeze(0).float())
            for j, second_im_path in enumerate(dataset_paths_list):
                # if j >= i:
                if second_im_path == im_path:
                    second_img_embedding = img_embedding
                else:
                    second_img = Image.open(second_im_path)
                    second_img = mtcnn(second_img)
                    second_img_embedding = model(second_img.unsqueeze(0).float())
                name1 = im_path[:-12]
                name2 = second_im_path[:-12]
                rdm[i, j] = cos(img_embedding, second_img_embedding)
                # rdm[j, i] = cos(img_embedding, second_img_embedding)
                print("cosimilarity between " + name1 + " and " + name2 + " is : " + str(rdm[i, j]))
        except:
            print(f'error calculating similarity for: {im_path}')

    return rdm


def thatcher_test(folder_path, model):
    row_indx = 2
    out = load_workbook("C:/Users/ynaho/PycharmProjects/galit_project/out_sheet.xlsx")
    thatcher_results_sheet = out.active
    for i, class_path in tqdm(enumerate(folder_path)):
        data_paths_list_up, data_paths_list_inv = load_data_for_thacher(class_path, 0)
        rdm = get_rdm_thacher(data_paths_list_up, data_paths_list_inv, model)
        thatcher_results_sheet.cell(row=row_indx, column=1).value = data_paths_list_up[0].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=2).value = data_paths_list_up[1].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=3).value = data_paths_list_inv[0].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=4).value = data_paths_list_inv[1].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=5).value = rdm[0]
        thatcher_results_sheet.cell(row=row_indx, column=6).value = rdm[1]
        thatcher_results_sheet.cell(row=row_indx, column=7).value = rdm[2]
        row_indx += 1
    out.save("C:/Users/ynaho/PycharmProjects/galit_project/out_sheet.xlsx")


def context_test(folder_path, model):
    row_indx = 2
    out = load_workbook("context_out_sheet.xlsx")
    context_results_sheet = out.active
    for i, class_path in tqdm(enumerate(folder_path)):
        rdm = get_context_test_results(class_path, model)
        context_results_sheet.cell(row=row_indx, column=1).value = data_paths_list[0].split('\\')[-1]
        context_results_sheet.cell(row=row_indx, column=2).value = data_paths_list[1].split('\\')[-1]
        context_results_sheet.cell(row=row_indx, column=3).value = rdm[0]
        context_results_sheet.cell(row=row_indx, column=4).value = rdm[1]
        context_results_sheet.cell(row=row_indx, column=5).value = rdm[2]
        row_indx += 1
    out.save("context_out_sheet.xlsx")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    args = parser.parse_args()
    data_paths_list = load_data_for_context_test(args.data_dir)
    # model = get_vgg_model(num_classes=len(data_paths_list))
    model = get_resnet_model()
    context_test(data_paths_list, model)
    # thatcher_test(data_paths_list, model)
    # rdm = get_rdm(data_paths_list, model)
    # plt.imshow(rdm)
    # plt.show()
