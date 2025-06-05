import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    '''生成具有高斯分布（正态分布）特性的数据样本'''
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims) # 生成的数据样本将具有的维度数
        self.bias = bias    # 可选参数，用于给生成的数据样本添加偏置
        self.scale = scale # 可选参数，用于缩放数据样本

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        '''
        n_points 指定每批生成多少数据点。
        b_size
        n_dims_truncated 是一个可选参数，指定在某个维度之后将数据截断（设置为零），这可以用于模拟降维的情况。
        seeds 是一个可选参数，如果提供了种子列表，将为每个批次生成可重复的数据。
        '''
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims) # 如果没有提供种子，直接生成标准高斯分布的随机数
        else:
            # print ('seeds = ', seeds) # [0, ..., 199], 每次一样。
            xs_b = torch.zeros(b_size, n_points, self.n_dims) # # 如果提供了种子，先初始化一个全零的张量
            generator = torch.Generator() # # 创建一个随机数生成器
            assert len(seeds) == b_size, (len(seeds), b_size) # 【确保提供的种子数与批次大小相等】
            for i, seed in enumerate(seeds): # 遍历种子列表
                generator.manual_seed(seed) # 设置生成器的种子
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator) # 生成bz中一个样本的高斯分布随机数

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 如果指定了截断维度，将超过该维度的数值置零
            xs_b[:, :, n_dims_truncated:] = 0
 
        # print ("xs_b[0][0] = ", xs_b[0][0]) # 每次一样。
        return xs_b # 返回生成的样本张量 [bz, n_points, n_dims]
