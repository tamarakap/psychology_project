import csv
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

my_path = os.path.abspath(os.path.dirname(__file__))
context_output_path = os.path.join(my_path, "../outputs/context_out_sheet.xlsx")
thatcher_output_path = os.path.join(my_path, "../outputs/thatcher_out_sheet.xlsx")
celebrities_folder_path = os.path.join(my_path, "../celebrities_for_context_test")
thatcher_images_path = os.path.join(my_path, "../img_by_identity")

normalize = det_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
composed_transforms = det_transforms.Compose([Rescale((160, 160)), normalize])

vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)


# resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

def get_resnet_model(pretrain='vggface2'):
    # options are: vggface2, casia-webface https://github.com/timesler/facenet-pytorch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
    #model = InceptionResnetV1(pretrained=pretrain).eval()
    return model


"""
def get_vgg_model(num_classes):
    model = vgg_model
    #model.fc = nn.Linear(in_features = 2048, out_features=num_classes+1, bias = True)
    model.classifier[6] = nn.Linear(in_features = 4096, out_features=num_classes+1, bias = True)
    return model
"""


def calculate_thacher_index(dif_up, dif_inv):
    return (dif_up - dif_inv) / (dif_up + dif_inv)


def get_image_embedding(img_path, model, rotate=False):
    img = Image.open(img_path)
    img.show()
    img = mtcnn(img)
    if rotate:
        img = torch.rot90(img, 2)

    # array = np.array(img)
    # img_tmp = Image.fromarray(np.uint8(array))
    # img_tmp.show()
    return model(img.unsqueeze(0).float())


def load_data_for_thacher(data_dir):
    data_paths_list = []
    data_paths_list2 = []
    data_dir = os.path.join(thatcher_images_path, data_dir)
    imgs = os.listdir(data_dir)

    for img in imgs[:2]:
        im_path = os.path.join(data_dir, img)
        data_paths_list.append(im_path)

    for img in imgs[2:]:
        im_path = os.path.join(data_dir, img)
        data_paths_list2.append(im_path)

    return data_paths_list, data_paths_list2


def load_data_for_context_test():
    return os.listdir(celebrities_folder_path)


def get_context_test_results(dataset_paths_list, model):
    rdm = np.zeros(2)
    dataset_paths_list = os.path.join(celebrities_folder_path, dataset_paths_list)
    imgs = os.listdir(dataset_paths_list)
    try:
        regular_img = os.path.join(dataset_paths_list, imgs[0])
        conext_img = os.path.join(dataset_paths_list, imgs[1])
        out_of_context_img = os.path.join(dataset_paths_list, imgs[2])

        regular_img_embedding = get_image_embedding(regular_img, model)
        conext_img_embedding = get_image_embedding(conext_img, model)
        out_of_context_img_embedding = get_image_embedding(out_of_context_img, model)

        rdm[0] = cos(regular_img_embedding, conext_img_embedding)
        rdm[1] = cos(regular_img_embedding, out_of_context_img_embedding)
    except:
        print(f'error calculating similarity')
    return rdm


def get_rdm_thacher(dataset_paths_list_up, dataset_paths_list_inv, model):
    rdm = np.zeros(3)
    try:
        img_path1 = dataset_paths_list_up[0]
        img_path2 = dataset_paths_list_up[1]

        img_embedding1 = get_image_embedding(img_path1, model)
        img_embedding2 = get_image_embedding(img_path2, model)

        rdm[0] = torch.cdist(img_embedding1, img_embedding2, 2)#cos(img_embedding1, img_embedding2)
    except:
        print(f'error calculating similarity')

    try:
        img_path1 = dataset_paths_list_inv[0]
        img_path2 = dataset_paths_list_inv[1]

        img_embedding1 = get_image_embedding(img_path1, model, rotate=True)
        img_embedding2 = get_image_embedding(img_path2, model, rotate=True)

        rdm[1] = torch.cdist(img_embedding1, img_embedding2, 2) #cos(img_embedding1, img_embedding2)
        #dist = (img_embedding1 - img_embedding2).pow(2).sum(3).sqrt()
    except:
        print(f'error calculating similarity')
    rdm[2] = calculate_thacher_index(rdm[0], rdm[1])
    return rdm


def thatcher_test(folder_path, model):
    row_indx = 2
    out = load_workbook(thatcher_output_path)
    thatcher_results_sheet = out.active
    for i, class_path in tqdm(enumerate(folder_path)):
        data_paths_list_up, data_paths_list_inv = load_data_for_thacher(class_path)
        rdm = get_rdm_thacher(data_paths_list_up, data_paths_list_inv, model)
        thatcher_results_sheet.cell(row=row_indx, column=1).value = data_paths_list_up[0].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=2).value = data_paths_list_up[1].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=3).value = data_paths_list_inv[0].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=4).value = data_paths_list_inv[1].split('\\')[-1]
        thatcher_results_sheet.cell(row=row_indx, column=5).value = rdm[0]
        thatcher_results_sheet.cell(row=row_indx, column=6).value = rdm[1]
        thatcher_results_sheet.cell(row=row_indx, column=7).value = rdm[2]
        row_indx += 1
    out.save(thatcher_output_path)


def context_test(folder_path, model):
    row_indx = 2
    out = load_workbook(context_output_path)
    context_results_sheet = out.active
    for i, class_path in tqdm(enumerate(folder_path)):
        rdm = get_context_test_results(class_path, model)
        context_results_sheet.cell(row=row_indx, column=1).value = rdm[0]
        context_results_sheet.cell(row=row_indx, column=2).value = rdm[1]
        row_indx += 1
    out.save(context_output_path)


# just paste this as running configuration:
# -test_type
# thatcher/context
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-test_type", "--test_type", dest="test_type", help="The type of test to perform")
    args = parser.parse_args()
    model = get_resnet_model()
    if args.test_type == "thatcher":
        thatcher_test(os.listdir(thatcher_images_path), model)
    elif args.test_type == "context":
        data_paths_list = load_data_for_context_test()
        context_test(data_paths_list, model)
