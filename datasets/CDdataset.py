import os
import einops
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from datasets.data_utils import *
import rasterio
from channel_nor import channel_normalization as cn

class load_and_split():
    def __init__(self, args):
        self.args = args
        self.data_path = os.path.join(self.args.data_path, self.args.data_name)

    def load_data(self):
        is_phase2 = ('林地二期' in self.data_path)  
        im1_name = '2015.tif' if is_phase2 else '2000.tif'
        im2_name = '2020.tif'
        with rasterio.open(os.path.join(self.data_path, im1_name)) as src:
            im1 = src.read()
        with rasterio.open(os.path.join(self.data_path, im2_name)) as src:
            im2 = src.read()

        last_chanel = im2[-1, :, :]  #取出第二图最后一层
        repeat_num = 2 if is_phase2 else 3
        repeat_channels = np.stack([last_chanel] * repeat_num, axis=0)
        im2 = np.concatenate((im2, repeat_channels), axis=0)

        im1 = np.where(im1 == 65535, 0, im1)
        im2 = np.where(im2 == 65535, 0, im2)   

        with rasterio.open(self.data_path + '/' + 'change.tif') as src:
            ground_truth = src.read(1)

        ground_truth = np.where(ground_truth == 255, 2, ground_truth)
        ground_truth = np.where(ground_truth == 0, 1, ground_truth)
        ground_truth = np.where(ground_truth == 65535, 0, ground_truth)

        num_band = im1.shape[0]

        im1 = torch.from_numpy(im1.astype(np.float32)).type(torch.FloatTensor)
        im2 = torch.from_numpy(im2.astype(np.float32)).type(torch.FloatTensor)
        ground_truth = torch.from_numpy(ground_truth.astype(np.float32)).type(torch.FloatTensor)

        half_window = int(self.args.patch_size // 2)
        pad = torch.nn.ReplicationPad2d(half_window)
        im1 = pad(im1.unsqueeze(0)).squeeze(0)
        im2 = pad(im2.unsqueeze(0)).squeeze(0)

        print(im1.max(), im1.min())
        print(im2.max(), im2.min())

        im1 = cn(im1)
        im2 = cn(im2)

        print(im1.max(), im1.min())
        print(im2.max(), im2.min())
        return im1, im2, ground_truth, num_band

    def cood_split(self):
        with rasterio.open(os.path.join(self.data_path, 'change.tif')) as src:
            ground_truth = src.read(1)
        ground_truth = np.where(ground_truth == 0, 1, ground_truth)
        ground_truth = np.where(ground_truth == 255, 2, ground_truth)
        ground_truth = np.where(ground_truth == 65535, 0, ground_truth)
        ground_truth = torch.from_numpy(ground_truth.astype(np.float32)).type(torch.FloatTensor)

        ground_truth = pad_with_zeros(ground_truth, self.args.patch_size)

        coord, label = get_train_coord(
            reference=ground_truth, patch_size=self.args.patch_size, data_name=self.args.data_name
        )  # 得到所有变化类与未变化类的坐标与真值
        test_cood = get_pred_coord(ground_truth, self.args.patch_size)

        # 划分训练集与测试集
        # assert len(coord) == len(test_cood) == len(label)
        print('数据总样本数:%d,其中变化样本个数：%d, 未变化样本个数: %d' % (len(label), label.count(1), label.count(0)))

        # if self.args.data_name == 'bayArea' or self.args.data_name == 'santaBarbara':
        #     train_coord, _, train_label, _ = train_test_split(
        #         coord, label, train_size=0.1, test_size=0.000001, stratify=label, random_state=42, shuffle=True)
        # else:
        #     train_coord, _, train_label, _ = train_test_split(
        #         coord, label, train_size=0.1, test_size=0.000001, stratify=label, random_state=42, shuffle=True)

        # coord 和 label 都是 list，需要转 numpy
        coords = np.array(coord)  # (N, 2)
        labels = np.array(label)  # (N,)

        N = len(labels)
        print("总样本:", N)

        # 想要抽取总量的 10%
        train_target = int(N * 0.1)

        # 固定随机数种子
        rng = np.random.default_rng(42)

        # 直接随机抽取 train_target 个样本
        train_indices = rng.choice(N, train_target, replace=False)

        # 构建训练集
        train_coord = coords[train_indices]
        train_label = labels[train_indices]

        # 剩余样本作为测试集（如果你需要）
        all_indices = np.arange(N)
        mask = np.ones(N, dtype=bool)
        mask[train_indices] = False
        test_indices = all_indices[mask]

        test_coord = coords[test_indices]
        test_label = labels[test_indices]

        print("训练集最终样本:", len(train_label))
        print("  其中类别 0:", (train_label == 0).sum())
        print("  其中类别 1:", (train_label == 1).sum())
        # print('训练集总数:%d,其中变化样本个数：%d'
        #        % (len(train_label), train_label.count(1)))
        return train_coord, train_label, test_cood

    def cropimg(im1, im2, index_coord, patch_size):
    # 确保返回的张量大小一致
       h, w = im1.shape[1:]
       margin = patch_size // 2
       y, x = index_coord
       im1_patch = im1[:, max(0, y - margin):min(h, y + margin + 1), max(0, x - margin):min(w, x + margin + 1)]
       im2_patch = im2[:, max(0, y - margin):min(h, y + margin + 1), max(0, x - margin):min(w, x + margin + 1)]

    # 填充到固定大小
       pad_h = patch_size - im1_patch.shape[1]
       pad_w = patch_size - im1_patch.shape[2]
       im1_patch = torch.nn.functional.pad(im1_patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
       im2_patch = torch.nn.functional.pad(im2_patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
       return im1_patch, im2_patch

class hyper_CDDataset(Data.Dataset):
    def __init__(self, args, coord, im1, im2, label=None):
        super(hyper_CDDataset, self).__init__()
        self.coord = coord
        self.label = label
        self.args = args
        self.im1 = im1
        self.im2 = im2

    def __getitem__(self, index):
        index_coord = self.coord[index]
        im1_patch, im2_patch = cropImg(self.im1, self.im2, index_coord, self.args.patch_size)
        
        # 断言确保张量大小一致
        assert im1_patch.shape == im2_patch.shape, f"im1_patch shape: {im1_patch.shape}, im2_patch shape: {im2_patch.shape}"
        
        if self.label is None:
            return {'data': [im1_patch, im2_patch]}
        label = self.label[index]
        return {'data': [im1_patch, im2_patch], 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.coord)



