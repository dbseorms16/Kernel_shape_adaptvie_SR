import os
import math
import argparse
import random
import logging
import imageio
import time
import pandas as pd
from copy import deepcopy

import torch
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data.baseline import loader, create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='temp')
    parser.add_argument('--degradation_type', type=str, default=None)
    parser.add_argument('--sigma_x', type=float, default=None)
    parser.add_argument('--sigma_y', type=float, default=None)
    parser.add_argument('--theta', type=float, default=None)
    args = parser.parse_args()
    if args.exp_name == 'temp':
        opt = option.parse(args.opt, is_train=False)
    else:
        opt = option.parse(args.opt, is_train=False, exp_name=args.exp_name)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if args.degradation_type is not None:
        if args.degradation_type == 'preset':
            opt['datasets']['val']['degradation_mode'] = args.degradation_type
        else:
            opt['datasets']['val']['degradation_type'] = args.degradation_type
    if args.sigma_x is not None:
        opt['datasets']['val']['sigma_x'] = args.sigma_x
    if args.sigma_y is not None:
        opt['datasets']['val']['sigma_y'] = args.sigma_y
    if args.theta is not None:
        opt['datasets']['val']['theta'] = args.theta
    
    if 'degradation_mode' not in opt['datasets']['val'].keys():
        degradation_name = ''
    elif opt['datasets']['val']['degradation_mode'] == 'set':
        degradation_name = '_' + str(opt['datasets']['val']['degradation_type'])\
                  + '_' + str(opt['datasets']['val']['sigma_x']) \
                  + '_' + str(opt['datasets']['val']['sigma_y'])\
                  + '_' + str(opt['datasets']['val']['theta'])
    else:
        degradation_name = '_' + opt['datasets']['val']['degradation_mode']
    folder_name = opt['name'] + '_' + degradation_name

    if args.exp_name != 'temp':
        folder_name = args.exp_name

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            pass
        elif phase == 'val':
            if '+' in opt['datasets']['val']['name']:
                raise NotImplementedError('Do not use + signs in test mode')
            else:
                val_set = create_dataset(dataset_opt, scale=opt['scale'],
                                     kernel_size=opt['kernel_size'])
                # val_set = loader.get_dataset(opt, train=False)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)

            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    modelcp = create_model(opt)

    with_GT = False if opt['datasets']['val']['mode'] == 'demo' else True

    pd_log = pd.DataFrame(columns=['PSNR_Bicubic', 'PSNR_Ours', 'SSIM_Bicubic', 'SSIM_Ours'])

    # Single GPU
    # PSNR_rlt: psnr_init, psnr_before, psnr_after
    psnr_rlt = 0
    ssim_rlt = 0
    avg_psnr_rlt= 0
    avg_ssim_rlt= 0
    # SSIM_rlt: ssim_init, ssim_after
    count = 0

    pbar = util.ProgressBar(len(val_set))
    for val_data in val_loader:
        train_folder = os.path.join('../test_results', folder_name )
        sub_train_folder = os.path.join(train_folder, opt['network_G']['which_model_G'])
        
        count += 1 
        if not os.path.exists(train_folder):
            os.makedirs(train_folder, exist_ok=False)
        if not os.path.exists(sub_train_folder):
            os.mkdir(sub_train_folder)

        
        modelcp.load_network(opt['path']['pretrain_model_G'], modelcp.netG)
        
        
        modelcp.feed_data(val_data, need_GT=with_GT)
        modelcp.test()

        if with_GT:
            model_start_visuals = modelcp.get_current_visuals(need_GT=True)
            hr_image = util.tensor2img(model_start_visuals['GT'], mode='rgb')
            lr_image = util.tensor2img(model_start_visuals['LQ'], mode='rgb')
            result = util.tensor2img(model_start_visuals['rlt'], mode='rgb')
            
            # Bic_LQ = F.interpolate(model_start_visuals['LQ'].unsqueeze(0), scale_factor=opt['scale'], mode='bicubic', align_corners=True)
            # Bic_LQ = util.tensor2img(Bic_LQ[0], mode='rgb')
            
            # psnr_rlt = util.calculate_psnr(result, hr_image)
            # ssim_rlt = util.calculate_ssim(result, hr_image)
            
            psnr_rlt = util.calculate_psnr(result, hr_image)
            ssim_rlt = util.calculate_ssim(result, hr_image)

            avg_psnr_rlt += psnr_rlt
            avg_ssim_rlt += ssim_rlt
        # modelcp.netG = deepcopy(model.netG)

        # Inner Loop Update
            
        # Save and calculate final image
        # print(os.path.join(maml_train_folder, '{:08d}.png'.format(idx_d)))
        # imageio.imwrite(os.path.join(sub_train_folder, 'Bicubic_{}'.format(val_data['filename'][0])), Bic_LQ)
        imageio.imwrite(os.path.join(sub_train_folder, 'SR_{}'.format(val_data['filename'][0])), result)
        # imageio.imwrite(os.path.join(sub_train_folder, 'HR_{}'.format(val_data['filename'][0])), hr_image)
        # imageio.imwrite(os.path.join(sub_train_folder, 'LR_{}'.format(val_data['filename'][0])), lr_image)

        if with_GT:
            # name_df = '{}'.format('HAN')
            # if name_df in pd_log.index:
            #     pd_log.at[name_df, 'PSNR_Ours'] = psnr_rlt
            #     pd_log.at[name_df, 'SSIM_Ours'] = ssim_rlt
            # else:
            #     pd_log.loc[name_df] = [psnr_rlt, ssim_rlt]

            # pd_log.to_csv(os.path.join('../test_results', folder_name, 'psnr_update.csv'))

            pbar.update('Test : I: {:.3f}/{:.4f} \t\t'
                            .format(psnr_rlt, ssim_rlt,
                                    ))
        else:
            pbar.update()

    if with_GT:
        psnr_rlt_avg = {}
        psnr_total_avg = avg_psnr_rlt / count 
        # Just calculate the final value of psnr_rlt(i.e. psnr_rlt[2])
        log_s = '# Validation # Bic PSNR: {:.4f}:'.format(psnr_total_avg)
        print(log_s)

        ssim_total_avg = avg_ssim_rlt / count 
        # Just calculate the final value of ssim_rlt(i.e. ssim_rlt[1])
        log_s = '# Validation # Bicubic SSIM: {:.4f}:'.format(ssim_total_avg)
        print(log_s)


    print('End of evaluation.')

if __name__ == '__main__':
    main()
