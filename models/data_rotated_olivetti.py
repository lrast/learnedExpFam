import numpy as np
import torch

from sklearn import datasets
from skimage.transform import rotate

from torch.utils.data import Dataset


def mask_images(images):
    """ Add a circular mask to prevent  """
    X_mask, Y_mask = np.meshgrid(np.arange(64) - 31.5, np.arange(64) - 31.5)

    for image in images:
        image[X_mask**2 + Y_mask**2 > 31.5**2] = 0.5

    return images


def rotate_images(images, angles):
    """ rotate the images """
    for i in range(images.shape[0]):
        images[i] = rotate(images[i], angles[i])

    return images


def train_test_split_by_identity(numbers=(300, 50, 50)):
    """ Consistant train-test split, with each face identity
        landing in a single split.
    """
    ntrain, nval, ntest = numbers
    data = datasets.fetch_olivetti_faces()

    if (ntrain % 10) or (nval % 10) or (ntest % 10):
        raise Exception("Face identities don't divide equally")

    np.random.seed(787)
    face_inds = np.random.permutation(40)

    def all_images(face_inds):
        img_inds = []
        for ind in face_inds:
            img_inds.append(np.where(data.target == ind)[0])

        return np.concatenate(img_inds)

    train_inds = all_images(face_inds[0:ntrain//10])
    val_inds = all_images(face_inds[ntrain//10:ntrain//10+nval//10])
    test_inds = all_images(face_inds[-(ntest//10):])

    return train_inds, val_inds, test_inds


class FaceDataset(Dataset):
    """ Dataset of rotated faces with calssification labels
    """
    def __init__(self, N_classes=4, split='train', **kwargs):
        super(FaceDataset, self).__init__()
        data = datasets.fetch_olivetti_faces()
        images = data['images']

        if 'numbers' in kwargs:
            tri, vali, testi = train_test_split_by_identity(kwargs['numbers'])
        else:
            tri, vali, testi = train_test_split_by_identity()

        if split == 'train':
            image_inds = tri
        if split == 'val' or split == 'validation':
            image_inds = vali
        if split == 'test':
            image_inds = testi

        N_images = image_inds.shape[0]

        angles = np.linspace(0, 2*np.pi, N_classes+1)[:-1]
        angle_ids = np.arange(N_classes)

        # repeat angles and images
        angles = np.repeat(angles, N_images)
        angle_classes = np.repeat(angle_ids, N_images)

        image_inds = np.tile(image_inds, N_classes)

        # process images
        images = images[image_inds]
        images = rotate_images(images, np.rad2deg(angles))
        images = mask_images(images)

        self.angles = angle_classes
        self.images = torch.as_tensor(images, dtype=torch.float).contiguous()

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch_images = self.images[idx, :, :]
        batch_angles = self.angles[idx]

        return {'image': batch_images, 'angle': batch_angles}
