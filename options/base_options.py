"""
This code is based from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import argparse
import os
# from util import util
import torch
import datetime
import json

import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.dataroot = '../../data/denoising'
        self.save_root = r'E:\TIM'
        self.checkpoints_dir = os.path.join(self.dataroot, 'checkpoints-flouroscopy_LDL')#self.checkpoints_dir = os.path.join(self.dataroot, 'checkpoints-flouroscopy_LDL')

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--prefix', type=str, default='',
            help='customized suffix: opt.savedir = prefix + opt.savedir')
        parser.add_argument('--suffix', type=str, default='',
            help='customized suffix: opt.savedir = opt.savedir + suffix')

        parser.add_argument('--multi_gpu',dest='multi_gpu', action='store_true',
            help='use all GPUs in machines')
        parser.add_argument('--no_multi_gpu', dest='multi_gpu', action='store_false',
            help='do not enable multiple GPUs')
        parser.set_defaults(multi_gpu=True)
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[],
            help='gpu ids: e.g. 0  0,1,2, 0,2. use [] for CPU')
        parser.add_argument('--device', type=str, default='cpu',
            help='CPU or GPU')
        parser.add_argument('--n_threads', type=int, default=4,
            help='number of threads for data loader to use, Default: 4')

        # directories to save the results
        parser.add_argument('--data_dir', default=self.dataroot,
            help='path to images')
        parser.add_argument('--save_root', default=self.save_root,
            help='path to images')
        parser.add_argument('--checkpoints_dir', type=str, default=self.checkpoints_dir,
            help='checkpoint directory')
        parser.add_argument('--savedir', type=str, default=None,
            help='models are saved here')

        parser.add_argument('--is_train', type=bool, default=True,
            help='phase')
        parser.add_argument('--load_epoch', type=str, default='latest',
            help='determine which epoch to load? set to latest to use latest cached model')

        # model
        parser.add_argument('--model', type=str, default='mfcnn2n2',
            choices=[
                'mfcnn', 'mfcnn2n', 'mfcnn2n2','mfattunet3', 'fl_unet_LDL_ver5', 'fl_4unet_LDL_ver5', 'fl_5unet_LDL_ver5','fl_model3_3',
                'estrnn', 'estrnn2n', 'unet', 'mfattunet3_2' ,'mfattunet3_2_rcnn','fl_2unet_LDL_ver5', 'fl_3unet_LDL_ver5','fl_3unet_LDL_ver55',
                # for attuent 
                'attunet', 'mfattunet3_3','fl_model3_3vae1_2_copy', 'fl_6unet_LDL_ver5', 'fl_unet_LDL_ver5_rcnn', 'fl_unet_LDL_ver5_rcnn_swt', 'fl_unet_LDL_ver5_rcnn_swt2', 'fl_unet_LDL_ver5_rcnn_swt3', 'fl_unet_LDL_ver5_rcnn_swt2_2', 'fl_unet_LDL_ver5_rcnn_swt3_2', 'mfatunet3_2_rcnn_vrt',
                # for LDL attention 부분
                'r2attunet', 'mfattunet3_wrap', 'mfattunet','fsaunet', 'fl_unet_LDL_ver5_rcnn_swt3_3','fl_unet_LDL_ver5_rcnn_swt3_3_l1','fl_unet_LDL_ver5_rcnn_swt3_3_rev','fl_unet_LDL_ver5_rcnn_swt3_4','mfattunet3_wrap_noise_c_loss','mfattunet3_wrap_first','fl_unet_LDL_ver5_rcnn_swt_fsaunet', 'ge_fl_unet_LDL_ver5_rcnn_swt3_3', 'fl_swt3_3ver2', 'dncnn', 'redcnn','mfattunet3_wrap_noise_loss', 'mfattunet3_wrap_noise', 'wganvgg','ablation',
                # for noise 추가한 first-step 
                'mfattunet3_2_noise_rcnn','fl_model3_3_copy','fl_model3_3_copy2','moving_model3_3_copy2_cb','moving_model3_3_copy2','moving_model3_3_copy2_l1','fl_model3_3vae','fl_model3_3vae1_2', 'fl_model3_3vae2','model1_vae', 'model1','model1_1','model2','r2net','fl_unet_LDL_3_3' ,'r2net2','fl_unet_LDL_ver5_rcnn_swt3_5','r2net3','ablation2','fastDVDnet','uddn',
                'UDCA_first', 'UDCA_first2', 'UDCA_first3', 'recursive_filter', 'UDCA_second', 'UDCA_second_2', 'UDCA_second_3' ,'UDCA_second2','UDCA_second2_2','UDCA_second2_3', 'UDCA_second3', 'UDCA_second3_2', 'UDCA_second3_3','r2net6', 'r2net7'
                'mfattunet3_2_noise_rcnn2', 'r2net5','mfattunet3_2_noise_rcnn3', 'mfattunet3_2_noise_rcnn4', 'fsunet_LDL_ver7',
                'mm_net_moving'
            ],      
            help='specify a model')

        # dataset parameters
        parser.add_argument('--datasets',  nargs='+', default=['moving700'],
            help='datasets for training')
        parser.add_argument('--test_datasets',  nargs='+', default=[],
            help='datasets for test')
        parser.add_argument('--test_dataset',  type=str, default='',
            help='dataset for test')
        parser.add_argument('--batch_size', type=int, default=32,
            help='input batch size')
        parser.add_argument('--n_channels', type=int, default=3,
            help='# of image channels')
        parser.add_argument('--rgb_range', type=float, default=255,
            help='maximum value of RGB or pixel value')
        parser.add_argument('--patch_size', type=int, default=80,
            help='size of patch')

        parser.add_argument('--n_inputs', type=int, default=5,
            help='number of image inputs to the network')
        parser.add_argument('--add_noise', default=False, action='store_true',
            help='add noise to clean image to generate noisy images')
        parser.add_argument('--noise', type=int, default=0,
            help='Gaussian Noise standard deviation, if set to 0, add random noise to make blind noise')

        parser.add_argument('--in_mem', action='store_true',
            help='load whole data into memory, default: False')
        parser.add_argument("--test_patches", dest='test_patches', action='store_true',
            help='divide image into patches when test')
        parser.add_argument('--test_image', dest='test_patches', action='store_false',
            help='test whole image when test')
        parser.set_defaults(test_patches=False)
        parser.add_argument('--patch_offset', type=int, default=15,
            help='size of patch offset')

        parser.add_argument('--resume', action='store_true',
            help='continue training: load the latest model')
        parser.add_argument('--ext', type=str, default='sep',
            help='dataset file extension (sep, img, reset)')

            
        # additional parameters
        parser.add_argument('--verbose', action='store_true',
             help='if specified, print more debugging information')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Set the basic options
        base_opt, _ = parser.parse_known_args()
        if self.is_train and base_opt.resume:
            self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt)
        elif not self.is_train and not base_opt.url:
            self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt)

        # modify model-related parser options
        model_name = base_opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        _, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        for dataset in base_opt.datasets:
            dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
        
        for dataset in base_opt.test_datasets:
            dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
            
        # save and return the parser
        self.parser = parser
        opt = parser.parse_args()
        
        if not self.is_train or opt.resume:
            opt.savedir = base_opt.savedir

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # Get savedir from model or select checkpoint dir
        if self.is_train and not opt.resume:
            # Set checkpoint directory to save
            set_savedir = models.get_savedir_setter(opt.model)
            opt.savedir = set_savedir(opt)
            self.save_options(opt)
            opt.log_file = os.path.join(opt.savedir, opt.log_file)
        elif  not self.is_train and opt.url:
            # when url is used, we need to specify all model parameters rather than loading from config.txt
            opt.savedir = os.path.join(opt.checkpoints_dir, opt.model)
        else:
            opt = self.load_options(opt)
        
        # print("savedir:", os.path.abspath(opt.savedir))
        if self.is_train:
            opt.exprdir = os.path.join(opt.savedir, 'expr')
            os.makedirs(opt.exprdir, exist_ok=True)
        else:
            opt.test_results_dir = os.path.join(opt.save_root, 'test-results-LDL', os.path.basename(opt.savedir))
            os.makedirs(opt.test_results_dir, exist_ok=True)
        # set gpu ids

        self.set_gpus(opt)

        self.print_options(opt)

        self.opt = opt
        return self.opt
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / train_opt.txt
        """
        

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        if self.is_train:
            file_name = os.path.join(opt.savedir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    """
    My methods
    """
    def save_options(self, opt):
        os.makedirs(opt.savedir, exist_ok=True)
        config_file = os.path.join(opt.savedir, "config.txt")
        with open(config_file, 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    def load_options(self, opt):
        # self.select_checkpoint_dir(opt)
        config_file = os.path.join(opt.savedir, "config.txt")
        with open(config_file, 'r') as f:
            # opt.__dict__ = json.load(f)
            saved_options = json.load(f)

        """
        Set parameters to be controlled
        """
        savedir = opt.savedir
        # test_random_patch = opt.test_random_patch
        resume = opt.resume
        for key in saved_options:
            if key in opt:
                # print("saved_options[{}]: {}".format(key, saved_options[key]))
                # print("opt[{}]: {}".format(key, opt.__dict__[key]))
                opt.__dict__[key] = saved_options[key]

        opt.savedir = savedir
        # opt.test_random_patch = test_random_patch
        opt.resume = resume
        return opt

    def select_checkpoint_dir(self, opt):
        print("checkpoint_dir:", os.path.abspath(opt.checkpoints_dir))
        dirs = os.listdir(opt.checkpoints_dir)

        for i, d in enumerate(dirs, 0):
            print("({}) {}".format(i, d))
        d_idx = input("Select directory that you want to load: ")

        path_opt = dirs[int(d_idx)]
        opt.savedir = os.path.abspath(os.path.join(self.checkpoints_dir, path_opt))
        print("savedir: {}".format(opt.savedir))

    def set_gpus(self, opt):
        n_gpu = torch.cuda.device_count()
        if opt.multi_gpu and len(opt.gpu_ids) == 0 and torch.cuda.is_available():
            opt.gpu_ids = list(range(torch.cuda.device_count()))
        elif len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            gpu_ids = opt.gpu_ids
            opt.gpu_ids = []
            for id in gpu_ids:
                if id >= 0 and id < n_gpu:
                    opt.gpu_ids.append(id)
            opt.gpu_ids = sorted(opt.gpu_ids)
        else:
            opt.gpu_ids = []
        
        if len(opt.gpu_ids) > 0:
            print("Enabling GPUs", opt.gpu_ids)
            if len(opt.gpu_ids) > 1:
                opt.multi_gpu = True
            else:
                opt.multi_gpu = False
            opt.device = "cuda:{}".format(opt.gpu_ids[0])
        else:
            print("No GPUs use")
            opt.device = "cpu"