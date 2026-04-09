import scipy.io as sio
import numpy as np
import scipy
import random
from sklearn.decomposition import PCA
import torch
from torchvision import transforms


def normalization(data):
    # 归一化为0-1
    img = torch.zeros_like(data)
    for b in range(img.size(0)):
        data_slice = data[b, :, :]
        img[b, :, :] = (data_slice-data_slice.min())/(data_slice.max()-data_slice.min())
    return img

def apply_pca(X, num_components):
    """apply pca to X and return new_X   对X应用pca并返回新的X"""
    new_X = np.reshape(X, (-1, X.shape[2]))  # 在不改变数据内容的情况下，改变一个数组的格式,(-1, X.shape[2])给定列数为X的通道数，自动计算出新数组的行数。
    # img.shape[0]：行数 img.shape[1]：列数 img.shape[2]：通道数
    pca = PCA(n_components=num_components, whiten=True)
    # n_components指定PCA降维后的特征维度数目。
    # whiten白化会去除变换信号中的一些信息(分量的相对方差尺度)，但在数据有比较强的相关性的假设下，有时可以提高下游估计器的性能。
    new_X = pca.fit_transform(new_X)  # 用X来训练PCA模型，同时返回降维后的数据newX。
    new_X = np.reshape(new_X, (X.shape[0], X.shape[1], num_components))
    return new_X


def std_norm(image):  # input tensor image size with CxHxW
    trans = transforms.Compose([
        transforms.Normalize(image.mean(dim=[1, 2]), image.std(dim=[1, 2]))
    ])   # (x - mean(x))/std(x) normalize to mean: 0, std: 1
    return trans(image)

def one_zero_norm(image):  # input tensor image size with CxHxW
    channel, height, width = image.shape
    data = image.reshape(channel, height * width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]

    data = (data - data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel

    return data.view(channel, height, width)

def norm(img):
    # img = np.asarray(img, dtype='float32')
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((m * n, -1))
    img = img / img.max()
    img_temp = np.sqrt(np.asarray((img ** 2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis=1)
    img_temp = img_temp.repeat(d, axis=1)
    img_temp[img_temp == 0] = 1
    img = img / img_temp
    img = np.reshape(img, (m, n, -1))
    return img


def get_train_coord(reference, patch_size, data_name):
    coord_list = []
    label_list = []
    height = reference.shape[0]
    width = reference.shape[1]
    border = patch_size // 2
    for h in range(height):
        for l in range(width):
            left = l - border  # 上下左右边界舍弃
            right = l + border
            above = h - border
            below = h + border
            if left < 0 or above < 0 or right >= width or below >= height:
                continue
            else:
                if data_name == "quyu1" or data_name == "quyu2" or data_name == "quyu3" or data_name == "quyu4" or data_name == "quyu5":
                    if reference[h, l] != 0:
                        coord_list.append((h, l))
                        label_list.append(int(reference[h, l]) - 1)
                elif data_name == "farm" or data_name == "river" or data_name == "third":
                    coord_list.append((h, l))
                    label_list.append(int(reference[h, l]))

    return coord_list, label_list

def get_pred_coord(reference, patch_size):
    coord_list = []
    border = patch_size // 2
    height = reference.shape[0]
    width = reference.shape[1]
    for h in range(height):
        for l in range(width):
            left = l - border  # 上下左右边界舍弃
            right = l + border
            above = h - border
            below = h + border
            if left < 0 or above < 0 or right >= width or below >= height:
                continue
            else:
                coord_list.append((h, l))

    return coord_list

def cropImg(im1, im2, coord, patch_size):
    upper_bound = int((patch_size + 1) / 2)
    lower_bound = int((patch_size - 1) / 2)
    h, w = coord
    im1_patch = im1[:, h - lower_bound:h + upper_bound, w - lower_bound:w + upper_bound]
    im2_patch = im2[:, h - lower_bound:h + upper_bound, w - lower_bound:w + upper_bound]

    # 确保返回的张量大小一致
    if im1_patch.shape != im2_patch.shape:
        # 填充较小的张量到较大的张量大小
        target_shape = max(im1_patch.shape, im2_patch.shape)
        padding_im1 = (0, target_shape[2] - im1_patch.shape[2], 0, target_shape[1] - im1_patch.shape[1])
        im1_patch = torch.nn.functional.pad(im1_patch, padding_im1, mode='constant', value=0)

        padding_im2 = (0, target_shape[2] - im2_patch.shape[2], 0, target_shape[1] - im2_patch.shape[1])
        im2_patch = torch.nn.functional.pad(im2_patch, padding_im2, mode='constant', value=0)
    return im1_patch, im2_patch


def pad_with_zeros(X, patch_size=5):
    """apply zero padding to X with margin  对带边距的X应用零填充"""
    margin = patch_size // 2
    if len(X.shape) == 3:
        new_X = np.zeros((X.shape[0], X.shape[1] + 2 * margin, X.shape[2]+ 2 * margin))  # 生成0填充数组，np.zeros（个，行，列）
        x_offset = margin
        y_offset = margin  # 补偿值
        new_X[: , x_offset:X.shape[1] + x_offset, y_offset:X.shape[2] + y_offset] = X  #
    else:
        new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin))  # 生成0填充数组，np.zeros（个，行，列）
        x_offset = margin
        y_offset = margin  # 补偿值
        new_X[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset] = X  #
    return new_X

def my_create_patches(data, window_size=5):
    """将(w,h,c)的图像裁剪成 (w*h,window_size,window_size,channel)的图像块，例如feature大小为(w, h, 432),
    裁剪后为(w*h,5,5,432)"""
    h, w, c = data.shape
    margin = int((window_size - 1) / 2)
    margin_padded_data = pad_with_zeros(data, margin=margin)
    # data_scale_to1 = margin_padded_data / np.max(margin_padded_data)
    patches_data = np.zeros((h * w, window_size, window_size, margin_padded_data.shape[-1])).astype(np.float32)
    patch_index = 0
    for i in range(margin, margin_padded_data.shape[0] - margin):
        for j in range(margin, margin_padded_data.shape[1] - margin):
            # patch的大小为(5,5,224)
            patch = margin_padded_data[i - margin:i + margin + 1, j - margin:j + margin + 1, :]
            patches_data[patch_index, :, :, :] = patch
            patch_index = patch_index + 1

    return patches_data

def my_load_dataset(data_path, data_name):
    im1 = sio.loadmat(data_path + data_name + "/" + "im1.mat")["im1"]
    im2 = sio.loadmat(data_path + data_name + "/" + "im2.mat")["im2"]
    ground_true = sio.loadmat(data_path + data_name + "/" + "ground_truth.mat")["ground_truth"]

    return im1, im2, ground_true

def oversample_weak_classes(X, y):
    """"balance the dataset by prforming oversample of weak classes (making each class have close labels_counts)
    样本均衡,将少样本的数据复制到与多样本一样"""
    unique_labels, labels_counts = np.unique(y, return_counts=True)  # 去除数组中的重复数字，并进行排序之后输出。

    print(unique_labels.shape)
    print(unique_labels)
    print(labels_counts.shape)
    print(labels_counts)
    max_count = np.max(labels_counts)  # 返回数组最大值
    labels_inverse_ratios = max_count / labels_counts
    # print(labels_inverse_ratios)
    # repeat for every label and concat 对每个标签和concat重复上述步骤
    print("labels_inverse_ratios:{}".format(labels_inverse_ratios))
    new_X = X[y == unique_labels[0], :].repeat(round(labels_inverse_ratios[0]), axis=0)
    new_Y = y[y == unique_labels[0]].repeat(round(labels_inverse_ratios[0]), axis=0)
    for label, labelInverseRatio in zip(unique_labels[1:], labels_inverse_ratios[1:]):
        cX = X[y == label, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        new_X = np.concatenate((new_X, cX))
        new_Y = np.concatenate((new_Y, cY))
    # 样本均衡后的数据是按便签分好类的，需要重新打乱顺序
    np.random.seed(seed=42)  # 用于指定随机数生成时所用算法开始的整数值
    rand_perm = np.random.permutation(new_Y.shape[0])  # 随机排列序列
    new_X = new_X[rand_perm, :]
    new_Y = new_Y[rand_perm]
    unique_labels, labels_counts = np.unique(new_Y, return_counts=True)

    return new_X, new_Y


def augment_data(X_1, X_2, alpha_range=(0.9, 1.1), beta=1 / 25):
    """augment the data by taking each patch and randomly performing
    a flip(up/down or right/left) or a rotation通过获取每个面片并随机执行翻转（上/下或右/左）或旋转来增加数据
"""
    if np.random.random() < 0.5:
        num = random.randint(0, 3)  # 用于生成一个指定范围内的整数。
        if (num == 0):
            X_1 = np.flipud(X_1)
            X_2 = np.flipud(X_2)# 用于翻转列表，将矩阵进行上下翻转
        if (num == 1):
            X_1 = np.fliplr(X_1)
            X_2 = np.fliplr(X_2)
        if (num == 2):
            no = random.randrange(-180, 180, 30)  # 从指定范围内，按指定基数递增的集合中获取一个随机数。
            X_1 = scipy.ndimage.interpolation.rotate(X_1, no, axes=(1, 0), reshape=False)
            X_2 = scipy.ndimage.interpolation.rotate(X_2, no, axes=(1, 0), reshape=False)
            # 旋转一个数组。（输入，旋转角度，两个轴定义旋转平面，reshape=ture时调整输出形状，以便输入数组完全包含在输出中，输出，
            # 样条插值的顺序，mode参数确定输入数组如何扩展到其边界之外）
        if (num == 3):
            alpha = np.random.uniform(*alpha_range)
            noise = np.random.normal(loc=0., scale=1.0, size=X_1.shape)
            X_1 = alpha * X_1 + beta * noise
            X_2 = alpha * X_2 + beta * noise
        # X_train = flipped_patch
    return X_1, X_2
def augment(x, alpha_range=(0.9, 1.1), beta=1 / 25, p=0.5):
    num = torch.rand(3)
    if num[0] < p:
        x = torch.fliplr(x)
    elif num[1] < p:
        x = torch.flipud(x)
    elif num[2] < p:
        alpha = np.random.uniform(*alpha_range)
        noise = torch.FloatTensor(np.random.normal(loc=0., scale=1.0, size=x.shape))
        x = alpha * x + beta * noise
    return x


