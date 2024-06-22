import numpy as np
from PIL import Image
from skimage import measure
import torchvision.transforms as transforms

def load_dataset(root, dataset, split_method):
    train_txt = root + '/img_idx/train.txt'
    test_txt = root + '/img_idx/test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    return train_img_ids, val_img_ids, test_txt

def load_dataset_eva(root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    return train_img_ids, val_img_ids, test_txt

def load_param(channel_size, backbone, blocks_per_layer=4):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]
    elif channel_size == 'all_48':
        nb_filter = [48, 48, 48, 48, 48]
    elif channel_size == 'all_32':
        nb_filter = [32, 32, 32, 32, 32]
    elif channel_size == 'all_16':
        nb_filter = [16, 16, 16, 16, 16]

    if backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]

    return nb_filter, num_blocks

# Define data augmentation transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
])

# Example of how to use data augmentation in the data loading function
def load_dataset_with_augmentation(root, dataset, split_method, transform=train_transforms):
    train_txt = root + '/img_idx/train.txt'
    test_txt = root + '/img_idx/test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.strip())  # Use strip() to remove newline characters
            line = f.readline()
    return train_img_ids, val_img_ids, test_txt, transform
#
# # Usage example
# root = '/path/to/dataset'
# dataset = 'your_dataset_name'
# split_method = 'your_split_method'
#
# train_img_ids, val_img_ids, test_txt, transform = load_dataset_with_augmentation(root, dataset, split_method)
#
# # Load and apply data augmentation to an image
# img_path = '/path/to/your/image.jpg'
# img = Image.open(img_path)
# img_augmented = transform(img)
#
# # Check the augmented image
# print(img_augmented.shape)  # Should output something like: torch.Size([3, 224, 224]) for a 3-channel image