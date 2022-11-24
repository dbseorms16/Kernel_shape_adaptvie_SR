import torch
import models.archs.classifier as Classifier
import models.archs.kernel_estimator as kernel_estimator
import models.archs.LRimg_estimator as lrimg_estimator


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        import models.archs.SRResNet_arch as SRResNet_arch
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        import models.archs.RRDBNet_arch as RRDBNet_arch
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'EDVR':
        import models.archs.EDVR_arch as EDVR_arch
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'], scale=opt['scale'])
    elif which_model == 'DUF':
        import models.archs.DUF_arch as DUF_arch
        if opt_net['layers'] == 16:
            netG = DUF_arch.DUF_16L(scale=opt['scale'], adapt_official=True)
        elif opt_net['layers'] == 28:
            netG = DUF_arch.DUF_28L(scale=opt['scale'], adapt_official=True)
        else:
            netG = DUF_arch.DUF_52L(scale=opt['scale'], adapt_official=True)

    elif which_model == 'BasicVSRplus':
        import models.archs.BasicVSRplus_arch as BasicVSRplus_arch
        netG = BasicVSRplus_arch.BasicVSRPlusPlus()
    
    elif which_model == 'HAN':
        import models.archs.han_arch as han_arch
        netG = han_arch.HAN(args=opt_net)

    elif which_model == 'SwinIR':
        import models.archs.swinir_arch as swinir_arch
        netG = swinir_arch.SwinIR(upscale=opt_net['scale'], in_chans=3, img_size=opt_net['training_patch_size'], window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    
    elif which_model == 'HAT':
        import models.archs.hat_arch as hat_arch
        netG = hat_arch.HAT()
        
    elif which_model == 'ARCNN':
        import models.archs.arcnn_arch as arcnn_arch
        netG = arcnn_arch.ARCNN(args=opt_net)

    elif which_model == 'QHAN':
        import models.archs.qhan as qhan
        netG = qhan.QHAN(args=opt_net)
        
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = Classifier.BasicClassifier(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Define Estimator
def define_E(opt):
    opt_net = opt['network_E']
    which_model = opt_net['which_model_E']
    use_bn = opt_net['use_BN']
    scale = opt['scale']

    if which_model == 'SFDN':
        netE = lrimg_estimator.DirectKernelEstimator_CMS(nf=opt_net['nf'])
    elif which_model == 'MFDN':
        netE = lrimg_estimator.DirectKernelEstimatorVideo(in_nc=opt_net['in_nc'], nf=opt_net['nf'], scale=scale)
    else:
        raise NotImplementedError('Estimator model [{:s}] not recognized'.format(which_model))
    return netE

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
