import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from data.meta_learner import preprocessing
from data import random_kernel_generator as rkg


class SETLQGTDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt, **kwargs):
        super(SETLQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': []}
        self.kernel_size = 21
        self.scale = self.opt['scale']
        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        assert self.paths_GT, 'Error: GT path is empty.'
        self.random_scale_list = [1]
        
        self.data_info['path_GT'].extend(self.paths_GT)
                
                # Generate kernel
        # if opt['degradation_mode'] == 'set':
        sigma_x = float(opt['sigma_x'])
        sigma_y = float(opt['sigma_y'])
        theta = float(opt['theta'])
        gen_kwargs = preprocessing.set_kernel_params(sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
        self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale, **gen_kwargs)
        self.gen_kwargs_l = [gen_kwargs['sigma'][0], gen_kwargs['sigma'][1], gen_kwargs['theta']]

        # elif opt['degradation_mode'] == 'preset':
        #     self.kernel_gen = rkg.Degradation(self.kernel_size, self.scale)
        #     if self.name.lower() == 'vid4':
        #         self.kernel_dict = np.load('F:/DynaVSR-master/pretrained_models/Vid4Gauss.npy')
        #     elif self.name.lower() == 'reds':
        #         self.kernel_dict = np.load('F:/DynaVSR-master/pretrained_models/REDSGauss.npy')
        #     else:
        #         raise NotImplementedError()

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        
    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                    ] if self.data_type == 'lmdb' else None
        filename = GT_path.split('\\')[-1]
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        # if self.opt['color']:  # change color space if necessary
        # img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # if self.opt['phase'] == 'train':

        H_s, W_s, _ = img_GT.shape
        
        if self.opt['phase'] == 'train': 
            patch_size = self.opt['patch_size']
            rnd_h = random.randint(0, max(0, H_s - patch_size))
            rnd_w = random.randint(0, max(0, W_s - patch_size))
            img_GT = img_GT[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        
            img_GT = util.augment([img_GT], self.opt['use_flip'],
                                            self.opt['use_rot'])
            
        img_GT = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = self.kernel_gen.apply(img_GT)

        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_GT.shape[2] == 3:
        #     img_GT = img_GT[:, :, [2, 1, 0]]
        #     img_LQ = img_LQ[:, :, [2, 1, 0]]


        return {'LQ': img_LQ, 'GT': img_GT, 'GT_path': GT_path, 'filename':filename}

    def __len__(self):
        return len(self.data_info['path_GT'])
