import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

HEIGHT = 224
WIDTH = 224

def align(img, landmarks):
    # This function takes in an image, detects the bounding boxes for the face or faces
    # in the image and then selects the face with the largest number of pixels.
    # for the largest face the eye centers are detected and the angle of the eyes with respect to
    # the horizontal axis is determined. It then provides this angle to the rotate_bound function
    # the rotate_bound function the rotates the image so the eyes are parallel to the horizontal axis

    left_eye = landmarks[1]
    right_eye = landmarks[0]
    lx, ly = left_eye
    rx, ry = right_eye
    dx = rx - lx
    dy = ry - ly
    tan = dy / dx
    theta = np.arctan(tan)
    theta = np.degrees(theta)
    img = rotate_bound(img, theta)
    return img


def crop_image(img, bbox):
    bbox[0] = 0 if bbox[0] < 0 else bbox[0]
    bbox[1] = 0 if bbox[1] < 0 else bbox[1]
    img = img[int(bbox[1]):  int(bbox[3]), int(bbox[0]): int(bbox[2])]
    return img


def rotate_bound(image, angle):
    # rotates an image by the degree angle
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def visualize(image, boxes, landmarks):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image)
    ax.axis('off')

    for box, landmark in zip(boxes, landmarks):
        ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
        ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
    fig.show()


if __name__ == '__main__':
    train_dir = r'C:\Users\shiri\Documents\School\faces\Data\VGG-Face2\data\train'
    cropped_aligned_dir = r'C:\Users\shiri\Documents\School\faces\Data\VGG-Face2\data\train_mtcnn_aligned_2000_300'
    classes = os.listdir(train_dir)

    mtcnn = MTCNN(post_process=False, image_size=HEIGHT, device=device)

    for c in tqdm(classes):
        c_path = os.path.join(train_dir, c)
        cropped_align_path = os.path.join(cropped_aligned_dir, c)
        if not os.path.isdir(cropped_align_path):
            os.mkdir(cropped_align_path)
        if c == '.DS_Store':
            continue

        imgs = os.listdir(c_path)
        for img in imgs:
            try:
                img_path = os.path.join(c_path, img)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                _, _, landmarks = mtcnn.detect(image, landmarks=True)

                aligned_face = align(image, landmarks[0])
                boxes, _, _ = mtcnn.detect(aligned_face, landmarks=True)
                cropped_align_faces = crop_image(aligned_face, boxes[0])
                cropped_align_faces = cv2.resize(cropped_align_faces, (HEIGHT, WIDTH))
                im = Image.fromarray(cropped_align_faces)
                im.save(os.path.join(cropped_align_path, f'{img}.jpg'))

            except:
                print (f'could not align image {img} from class {c}')
                continue
