import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug.augmenters as iaa  # 导入iaa

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    # def __call__(self, sample):
    #     image, label = sample['image'], sample['label']
    #
    #     if random.random() > 0.5:
    #         image, label = random_rot_flip(image, label)
    #     elif random.random() > 0.5:
    #         image, label = random_rotate(image, label)
    #     x, y = image.shape
    #     if x != self.output_size[0] or y != self.output_size[1]:
    #         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
    #         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
    #     image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    #     label = torch.from_numpy(label.astype(np.float32))
    #     sample = {'image': image, 'label': label.long()}
    #     return sample
    def __call__(self, sample):
        # 具体来说，该函数首先从输入样本中提取图像和标签，
        image, label = sample['image'], sample['label']
        # 根据一定概率对图像进行随机旋转和翻转操作，以增强数据的多样性。
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, _ = image.shape
        # 然后，它检查图像的大小是否与指定的输出大小相同。
        if x != self.output_size[0] or y != self.output_size[1]:
            # 如果不同，它使用三次插值重新采样图像，并使用最近插值重新采样标签，使它们与输出大小匹配。
            # 需要注意的是，这里的zoom函数与前面稍有不同，它增加了一个维度，以便同时处理图像和标签。
            # 在这个维度上，它不进行任何插值操作。由于这个维度被设置为1，因此在进行转换时并不会改变张量的形状。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # 接着，它将增强后的图像和标签转换为PyTorch张量
        # 另外，这里的permute函数用于重新排列张量的维度，以符合PyTorch的要求。它将图像的维度从(H, W, C)转换为(C, H, W)。
        image = torch.from_numpy(image.astype(np.float32))
        # 将图像的维度从(H, W, C)转换为(C, H, W)，以符合PyTorch的要求。
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        # 最后，它将增强后的图像和标签打包为一个字典，作为函数的输出。
        sample = {'image': image, 'label': label.long()}
        return sample


# class Synapse_dataset(Dataset):
#     # def __init__(self, base_dir, list_dir, split, transform=None):
#     #     self.transform = transform  # using transform in torch!
#     #     self.split = split
#     #     self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#     #     self.data_dir = base_dir
#     #
#     # def __len__(self):
#     #     return len(self.sample_list)
#     #
#     # def __getitem__(self, idx):
#     #     if self.split == "train":
#     #         slice_name = self.sample_list[idx].strip('\n')
#     #         data_path = os.path.join(self.data_dir, slice_name+'.npz')
#     #         data = np.load(data_path)
#     #         image, label = data['image'], data['label']
#     #     else:
#     #         vol_name = self.sample_list[idx].strip('\n')
#     #         filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#     #         data = h5py.File(filepath)
#     #         image, label = data['image'][:], data['label'][:]
#     #
#     #     sample = {'image': image, 'label': label}
#     #     if self.transform:
#     #         sample = self.transform(sample)
#     #     sample['case_name'] = self.sample_list[idx].strip('\n')
#     #     return sample
class Synapse_dataset(Dataset):
    # 类的初始化函数__init__有四个参数：
    def __init__(self, base_dir, list_dir, split, transform=None):
        # base_dir：数据集所在的文件夹路径
        # list_dir：数据集列表文件所在的文件夹路径
        # split：数据集分割（训练集、验证集、测试集）的名称

        # transform：对数据进行预处理的函数，这里使用了PyTorch提供的transform函数
        self.transform = transform  # using transform in torch!
        self.split = split

        # self.sample_list存储了数据集中所有数据的文件名，这些文件名是从列表文件中读取的，
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),                             # 随机剪裁图片比例 0——0.1
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),               # 添加高斯噪声，对于50%的图片,这个噪采样对于每个像素点指整张图片采用同一个值；剩下的50%的图片，对于通道进行采样(一张图片会有多个值)；改变像素点的颜色(不仅仅是亮度)
            iaa.GaussianBlur(sigma=(1.0)),                              # 高斯模糊
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),            # 增强或减弱图片的对比度
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),           # 缩放变换
            iaa.Affine(rotate=(-40, 40)),                               # 旋转
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)





    def __len__(self):
        return len(self.sample_list)
    # len方法返回样本的数量。


    # 这是一个PyTorch数据集类的实现，包括了__getitem__方法用于从数据集中获取单个样本。
    #
    # 如果数据集划分为“train”，则根据样本列表中的样本名称加载相应的图像和标签数据。
    # 具体地，首先使用strip()方法去掉样本名称字符串两端的空格和换行符，
    # 然后通过os.path.join()方法组合数据目录和样本名称得到数据文件的路径。
    # 使用np.load()函数读取.npz格式的数据文件，并将其中的image和label数据分别赋值给image和label变量。
    #
    # 如果数据集不是“train”，则根据样本列表中的体积名称加载相应的图像和标签数据。
    # 具体地，首先使用strip()方法去掉体积名称字符串两端的空格和换行符，
    # 然后通过str.format()方法组合数据目录和体积名称得到数据文件的路径。
    # 使用h5py.File()函数读取.npy.h5格式的数据文件，
    # 并将其中的image和label数据分别赋值给image和label变量。
    # 无论是哪种数据集，最终将image和label组成字典类型的样本，并使用self.transform对其进行预处理。
    # 最后，将样本的名称加入字典中并返回。

    # 包括了__getitem__方法用于从数据集中获取单个样本。
    def __getitem__(self, idx):
        # 如果数据集划分为“train”，则根据样本列表中的样本名称加载相应的图像和标签数据。
        if self.split == "train":
            # 首先使用strip()方法去掉样本名称字符串两端的空格和换行符
            slice_name = self.sample_list[idx].strip('\n')
            # 然后通过字符串拼接形成数据文件的路径。
            data_path = self.data_dir + "/" + slice_name + '.npz'
            # 使用np.load()函数读取.npz格式的数据文件
            data = np.load(data_path)
            # 将其中的image和label数据分别赋值给image和label变量。
            image, label = data['image'], data['label']

        # 如果数据集不是“train”，则同样根据样本列表中的样本名称加载相应的图像和标签数据，
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            image, label = data['image'], data['label']
            # 但是在加载后，将image和label转换为PyTorch张量，
            image = torch.from_numpy(image.astype(np.float32))
            # # 并使用permute()方法重新排列image的维度为(channel, height, width)的形式。
            image = image.permute(2, 0, 1)
            label = torch.from_numpy(label.astype(np.float32))

        # 无论是哪种数据集，最终将image和label组成字典类型的样本
        sample = {'image': image, 'label': label}

       # 使用self.transform对其进行预处理。
        if self.transform:
            sample = self.transform(sample)
        # 最后，将样本的名称加入字典中并返回。
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
