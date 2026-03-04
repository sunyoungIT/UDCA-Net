from audioop import mul
import os
import datetime
from reprlib import recursive_repr

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel
from models.convs import common
from torch.nn.modules.utils import _pair, _single
from models.common.unet import create_unet
#from models.mfattunet3 import create_unet2 as create_mfunet3
#from models.common.LDL_loss import get_refined_artifact_map
from models.common.LDL_loss2 import get_refined_artifact_map
from models.common.LDL_loss5 import get_refined_artifact_map as get_refined_artifact_map5
from models.common.recursive_filter import RecursiveFilter
from models.mfattunet3_wrap import create_model as create_mfunet3
from models.mfattunet3_2_rcnn import create_model as create_mfrcnnunet3
from models.convs.wavelet import SWTForward, serialize_swt, SWTInverse, unserialize_swt
from models.mfattunet3_wrap_noise import create_model as create_mfattunet3_wrap_noise
from models.mfattunet3_2_noise_rcnn import create_model as create_mfattunet3_noise
url = {
    # 'n5m32g32n5': 'http://gofile.me/4u1bp/95jMJyKDt',
    'n3m32g32n5': 'https://www.dropbox.com/s/9gxcjy706ho4qqt/epoch_best_n0003_loss0.00032845_psnr34.8401.pth?dl=1'
}
def create_qenet(opt):
    if opt.qenet == 'mfselfattn':

        #qenet = create_mfunet3(opt)  # -> #1 짝꿍 # 1853, 1856
        #qenet = create_mfattunet3_wrap_noise(opt) # -> 3 짝꿍
        qenet = create_mfattunet3_noise(opt)

        # for moving700 -> fist step : mfattunet3_wrap
        #pretrained_path = '../../data/denoising/checkpoints-flouroscopy_LDL/moving700-20230116-2152-mfattunet3_wrap-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/pth/epoch_30_n0030_loss0.00030289_psnr35.2860.pth'
        #pretrained_path = '../../data/denoising/checkpoints-flouroscopy_LDL/moving700-20230131-1623-mfattunet3_wrap_noise-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/pth/epoch_best_n0048_loss0.00083241_psnr30.8716.pth' # -> #3짝꿍
        
        #pretrained_path = '../../data/denoising/checkpoints-flouroscopy_LDL/moving700-20230324-0316-mfattunet3_2_noise_rcnn-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/pth/epoch_150_n0150_loss0.00036960_psnr34.5918.pth'
        # 원래 moving00 으로 first step 에서 spynet 뺀거 
        pretrained_path = '../../data/denoising/checkpoints-flouroscopy_LDL/moving700-20230402-2019-mfattunet3_2_noise_rcnn-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/pth/epoch_170_n0170_loss0.00040239_psnr34.2359.pth'
        # for genoray 
        #pretrained_path = '../../data/denoising/checkpoints-flouroscopy_LDL/genoray-20230116-2313-mfattunet3_wrap-n_inputs5-ms_channels32-growth_rate32-n_denselayers5/pth/epoch_best_n0100_loss0.00005316_psnr43.1262.pth'
    else:
        raise ValueError('specify qenet')
        
    checkpoint = torch.load(pretrained_path)
    qenet.load_state_dict(checkpoint['mfattunet3']) # -> #1 짝꿍
    #qenet.load_state_dict(checkpoint['mfattunet'])# -> #2 짝꿍

    return qenet

class R2Net5(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        # n_inputs is set in base options
        # n_channels is set in dataset options
        
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        
        parser.add_argument('--n_dense_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        
        parser.add_argument('--n_denseblocks', type=int, default=2,
            help='number of layers of dense blocks')
        parser.add_argument('--qenet', type=str, default='mfselfattn',
            help='get_refined_map sigma values')
        parser.add_argument('--sig', type=float, default=0.005, # n2m
            help='n2m size of patch')
        parser.add_argument('--patch_size2', type=int, default=120, # n2m
                help='n2m size of patch')
        parser.add_argument('--alpha', type=float, default=0.08, # n2m
            help='n2m size of patch')
        parser.add_argument('--beta', type=float, default=1, # n2m
            help='n2m size of patch')
        parser.add_argument('--art_beta', type=float, default=1, # n2m
            help='n2m size of patch')
        parser.add_argument('--patch_number', type=int, default=10, # n2m
                help='number of patch') 
        parser.add_argument('--w', type=float, default=0.2, # for recursive
                help='recursive parameter')
        parser.add_argument('--ms_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        parser.add_argument('--growth_rate', type=int, default=32,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_denselayers', type=int, default=5,
            help='number of layers in dense block')
        # n_denseblocks is currently is not used
        # parser.add_argument('--n_denseblocks', type=int, default=8,
        #     help='number of layers of dense blocks')

        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        
        parser.set_defaults(depth=5)
        parser.set_defaults(patch_size=160)
        parser.add_argument('--backbone', type=str, default='unet',
            choices=['unet'],
            help='backbone model'
        )
        opt, _ = parser.parse_known_args()
        if opt.backbone == 'unet':
            parser.add_argument('--bilinear', type=str, default='bilinear',
                help='up convolution type (bilineaer or transposed2d)')

        if is_train:
            parser = parse_perceptual_loss(parser)
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)
            parser.set_defaults(valid_ratio=0.1)
        else:
            parser.set_defaults(test_patches=False)

        return parser

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model
        model_opt = model_opt + "-n_inputs" + str(opt.n_inputs)
        model_opt = model_opt + "-ms_channels" + str(opt.ms_channels)
        model_opt = model_opt + "-growth_rate" + str(opt.growth_rate)
        model_opt = model_opt + "-n_denselayers" + str(opt.n_denselayers)
        # model_opt = model_opt + "-n_denseblocks" + str(opt.n_denseblocks)
        if opt.perceptual_loss is not None:
            model_opt = model_opt + '-perceptual_loss' + '-' + opt.perceptual_loss

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        savedir = os.path.join(opt.checkpoints_dir, model_opt)
        return savedir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.swt_lv = 1
        self.swt = SWTForward(J=self.swt_lv, wave=opt.wavelet_func).to(self.device)
        self.iswt = SWTInverse(wave=opt.wavelet_func).to(self.device)
        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        if self.perceptual_loss and self.is_train:
            self.loss_name = ['content_loss', 'style_loss']
        else:
            self.loss_name = ['content_loss']

        self.model_names = ['netG']
        self.var_name = ['x', 'out', 'target']
        self.motion = create_model(opt).to(self.device)
        # Create model
        self.netG = create_unet(opt).to(self.device)
        self.nc = opt.n_channels
        self.n_inputs = opt.n_inputs
        self.beta = opt.beta
        self.art_beta = opt.art_beta
        self.sig = opt.sig
        w = opt.w
        # Define losses and optimizers
        if self.is_train:
            self.criterionL1 = nn.L1Loss()
            if opt.content_loss == 'l1':
                #print("opt.content_loss using.....")
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.perceptual_loss_criterion = PerceptualLoss(opt)
            self.model_names = ['qenet','netG']

            self.recur = RecursiveFilter(w)
            print("using w====> ", w)
            self.qenet = create_qenet(opt).to(self.device)
            self.set_requires_grad([self.qenet], False)
            self.optimizer_names = ['optimizerG']
            self.optimizerG = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0
            )
            self.optimizers.append(self.optimizerG)
            

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()
        self.n_inputs = opt.n_inputs

        url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        self.ix = torch.cat((self.x[:,:self.n_inputs//2], self.x[:,self.n_inputs//2+1:]), dim=1)
        #print("self.ix -> recursive input shaep :", self.ix.shape)
        #self.tx = self.ix.unsqueeze(1) # 최종 사용한다고 했을때 이것만 있음 되었음. # 1856 , 1853

        # for mfattunet3_noise_wrap버전        
        self.tx = self.x.clone()
        self.tx[:,self.n_inputs//2] = self.tx[:,self.n_inputs//2]+ self.tx[:,self.n_inputs//2].normal_(std=0.1)
        #self.tx = self.tx.unsqueeze(1)
        #print("self.tx -> qenet shaep :", self.tx.shape)

        self.recurx = self.x.clone()
        self.recurx = self.recurx.unsqueeze(1)
        self.ct =self.x[:, self.n_inputs//2:self.n_inputs//2+1]
        #print("U-net self.ct.shape ->:", self.ct.shape)
        #print("self.tx.shape : ", self.tx.shape)
        if input['target'] is not None:
            target = input['target'].to(self.device)
            
            self.target = target[:, self.n_inputs//2:self.n_inputs//2+1]
            
        bs, n, h, w = self.x.shape
        
    def forward(self):
        #print("self.ct.shape :", self.ct.shape)
        self.out = self.netG(self.ct) # u-net 
        # self.qe_out = self.qenet(self.ix.detach())

    def forward_train(self):
        self.out = self.netG(self.ct)          # [bs, 1, h, w]
        # print("out.shape", self.out.shape)
        self.pre_out = self.qenet(self.tx.detach())
        # print("pre_out.shape", self.pre_out.shape)
        self.motion_out = self.motion(self.recurx.detach())
        #print(self.motion_out.shape) # (bs, 1, 5, h, w) or (bs,5 ,1, h,)
        # print("motion_out.shape", self.motion_out.shape)
        self.recur_out = self.recur(self.motion_out.detach())
        # print("recur_out.shape", self.recur_out.shape)
        #print("self.out.shape : ", self.out.shape)
        #print("self.pre_out.shape : ", self.pre_out.shape)
        #print("self.recur_out.shape :", self.recur_out.shape)
        self.qe_out = torch.mul(0.5*self.out, self.recur_out)
        #print("qe_out.shape", self.qe_out.shape)
        self.qe_out2 = torch.mul(self.out, self.recur_out)
        #recur_out 에 swt 적용 
        self.recur_out_swt = self.swt(self.recur_out)
        self.recur_out_swt = serialize_swt(self.recur_out_swt)
        self.high_recur_out_swt = self.recur_out_swt[:,self.nc:]
        self.out_swt = self.swt(self.out)
        self.out_swt = serialize_swt(self.out_swt)
        self.high_out_swt = self.out_swt[:,self.nc:]
        self.pixel_weight = get_refined_artifact_map(self.qe_out, self.ct, self.pre_out, 7, self.sig)
        
        # test_dir = r'D:/data'
        # import os
        # a=torch.mul(self.pixel_weight, self.high_out_swt)
        # b=torch.mul(self.pixel_weight, self.high_recur_out_swt)
        # from skimage.io import imsave
        # recursive_dir = os.path.join(test_dir, 'here+++++')
        # out = a[:,0:1,:,:].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # recursive_path = os.path.join(recursive_dir + 'aa_qe_out.tiff')
        # out2 = b[:,0:1,:,:].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # recursive_path2 = os.path.join(recursive_dir + 'aa_qeout_2.tiff')
        # out3 = self.pre_out.detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # recursive_path3 = os.path.join(recursive_dir + 'aa_pre_out.tiff')
        # imsave(recursive_path, out)
        # imsave(recursive_path2, out2)
        # imsave(recursive_path3, out3)
        # print(t.shape)
        
        #self.pixel_weight = get_refined_artifact_map(self.qe_out, self.out, self.recur_out, 7, self.sig)
        
        

        # print("self.pre_out .shape :", self.pre_out.shape)
        
      
    def backward(self):
        #self.artif_loss = self.content_loss_criterion(torch.mul(self.pixel_weight, self.out), torch.mul(self.pixel_weight, self.recur_out))
        if self.perceptual_loss:

            self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.ct, self.out)
            #print(self.content_loss)
            #print("self.style loss : ", self.style_loss)
            self.loss = self.content_loss + self.style_loss #+ 1000*self.artif_loss
        else:
            self.loss_gq = self.criterionL1(self.out, self.recur_out)
            self.loss_pre = self.criterionL1(self.out, self.pre_out)
            self.artif_loss = self.criterionL1(torch.mul(self.pixel_weight, self.high_out_swt), torch.mul(self.pixel_weight, self.high_recur_out_swt))
            print("--------- loss_gq : ", self.loss_gq)
            print("--------- pretrained loss :", self.loss_pre)
            print("--------- artfact loss : ",self.artif_loss)
            print("self.loss_gq + self.art_beta * self.artif_loss + self.beta*self.loss_pre :", self.loss_gq + self.art_beta * self.artif_loss + self.beta*self.loss_pre)
        self.loss = self.beta*self.loss_gq + self.art_beta * self.artif_loss + self.loss_pre
        self.loss.backward()
        
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizerG.zero_grad()
        self.forward_train()
        self.backward()
        self.optimizerG.step()
    

    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        if self.perceptual_loss:
            #print("Content Loss: {:.8f}, Style Loss: {:.8f}, Artifact Loss: {:.11f}".format(
            #    self.content_loss, self.style_loss, self.artif_loss)
            #)
            print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
                self.content_loss, self.style_loss)
            )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )

def create_model(opt):
    return FSAUNetModel(opt)

class FirstBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pa_frames=2):
        super(FirstBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pw = ParallelWarping(out_channels, out_channels, pa_frames=pa_frames)

    def forward(self, x):
        x = self.double_conv(x)
        # (bs, c, n, h, w) -> (bs, n, c, h, w)
        #x = x.transpose(1, 2)
        #x = self.pw(x, flows_backward, flows_forward)

        # (bs, c, n, h, w)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(single_conv, self).__init__()
        m_body = []
        m_body.append(nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        if bn: m_body.append(nn.BatchNorm3d(out_ch))
        m_body.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*m_body)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            single_conv(in_channels, out_channels, bn=bn),
            single_conv(out_channels, out_channels, bn=bn)
        )
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print('x1.pad.shape:', x1.shape)
        # print('x2.shape:', x2.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.
    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device), torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    # print('grid.shape:', grid.shape)
    # print('flow.shape:', flow.shape)
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4': # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest', padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


class ParallelWarping(nn.Module):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.
    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 in_dim,
                 dim,
                 pa_frames=2,
                 deformable_groups=16,
                 reshape=None,
                 max_residue_magnitude=10,
                 ):
        super(ParallelWarping, self).__init__()
        self.pa_frames = pa_frames

        # parallel warping
        self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                             max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
        self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        x_backward, x_forward = getattr(self, f'get_aligned_feature_{self.pa_frames}frames')(x, flows_backward, flows_forward)
        x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x


    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        # print("get_aligned x.shape:", x.shape)
        # print('get_aligned x_backward.shape:', x_backward[0].shape)
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            # print(f'get_aligned {i}flow.shape:', flow.shape)
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            # print(f'get_aligned {i}x_i_warped.shape:', x_i_warped.shape)
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 4 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = torch.zeros_like(x[:, n - 2, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_backward.insert(0,
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = torch.zeros_like(x[:, 1, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_forward.append(
                self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 6 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = torch.zeros_like(x[:, -1, ...])
                flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = torch.zeros_like(x[:, -1, ...])
                flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
            x_backward.insert(0,
                              self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                             x[:, i - 3, ...], [flow1, flow2, flow3]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = torch.zeros_like(x[:, 0, ...])
                flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = torch.zeros_like(x[:, 0, ...])
                flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
            x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
            x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
                                            x[:, i + 1, ...], [flow1, flow2, flow3]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    # def forward(self, x):
    #     out = self.conv_offset(x)
    #     o1, o2, mask = torch.chunk(out, 3, dim=1)
    #     offset = torch.cat((o1, o2), dim=1)
    #     mask = torch.sigmoid(mask)
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.
    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)

class Mlp_GEGLU(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".
    Args:
        x: (B, D, H, W, C)
    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Linear(in_features, hidden_features)
        self.fc12 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pa_frames=2):
        super(DownBlock, self).__init__()
        self.down = Down(in_channels, out_channels)
        self.pw = ParallelWarping(out_channels, out_channels, pa_frames=pa_frames)

    def forward(self, x, flows_backward, flows_forward):
        bs, c, n, h, w = x.shape
        #print('down x.shape:', x.shape)
        x = self.down(x)

        # (bs, c, n, h, w) -> (bs, n, c, h, w)
        x = x.transpose(1, 2)
        x = self.pw(x, flows_backward, flows_forward)

        # (bs, c, n, h, w)
        return x #.transpose(1, 2)

class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, pa_frames=2):
        super(UpBlock, self).__init__()
        self.up = Up(in_channels, out_channels, bilinear=bilinear)
        self.pw = ParallelWarping(out_channels, out_channels, pa_frames=pa_frames)

    def forward(self, x1, x2, flows_backward, flows_forward):
        x = self.up(x1, x2)

        x = x.transpose(1, 2)
        x = self.pw(x, flows_backward, flows_forward)

        # (bs, c, n, h, w)
        return x # .transpose(1, 2)

class SpyNet(nn.Module):
    """SpyNet architecture.
    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class Out2dConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out2dConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class FSAUNetModel(nn.Module):
    def __init__(self, opt):
        super(FSAUNetModel, self).__init__()
        n_channels = opt.n_channels
        bilinear = opt.bilinear
        w = opt.w
        self.pa_frames = 2
        # chs = [64, 128, 128, 256, 256, 512, 512, 256, 256, 128, 128, 64]
        chs = [32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32]
        #chs = [32,64,128,256]
        growth_rate = opt.growth_rate
        # self.inc = DoubleConv(n_channels, 64)
        # self.conv_first = nnConv3d(n_channels*(1+2*4), 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        factor = 2 if bilinear else 1
        self.conv_first = FirstBlock(n_channels*(1+2*4), 1, pa_frames=self.pa_frames)
        self.down1 = DownBlock(chs[0], chs[1], self.pa_frames)
        self.down2 = DownBlock(chs[2], chs[3], self.pa_frames)
        self.down3 = DownBlock(chs[4], chs[5] // factor, self.pa_frames)


        # multi-scale
        ms_channels = opt.ms_channels
        self.nc = n_channels
        self.n_inputs = opt.n_inputs
        dense_in_channels = n_channels * ms_channels * 3 * self.n_inputs
        #multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_inputs)]
        #self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.out2d = Out2dConv(growth_rate, n_channels)
        self.recur = RecursiveFilter(w)
        #self.unet = MFR2AttU_Net(dense_in_channels, growth_rate)
        # self.down3 = DownBlock(256, 512, self.pa_frames)
        # self.down4 = DownBlock(512, 1024 // factor, self.pa_frames)
    
        # self.convs = nn.Sequential(
        #     single_conv(1024, 1024),
        #     single_conv(1024, 1024),
        #     single_conv(1024, 1024),
        #     single_conv(1024, 1024),
        #     single_conv(1024, 1024),
        #     single_conv(1024, 1024),
        # )

        # self.up1 = UpBlock(1024, 512 // factor, bilinear, self.pa_frames)
        # self.up2 = UpBlock(512, 256 // factor, bilinear, self.pa_frames)
        # self.up3 = UpBlock(256, 128 // factor, bilinear, self.pa_frames)
        # self.up4 = UpBlock(128, 64, bilinear, self.pa_frames)
        # self.up1 = UpBlock(1024, 512 // factor, bilinear, self.pa_frames)
        self.up1 = UpBlock(chs[6], chs[7] // factor, bilinear, self.pa_frames)
        self.up2 = UpBlock(chs[8], chs[9] // factor, bilinear, self.pa_frames)
        self.up3 = UpBlock(chs[10], chs[11], bilinear, self.pa_frames)
        self.outc = OutConv(chs[11], n_channels)

        self.spynet = SpyNet(f'{opt.data_dir}/spynet/spynet.pth', [2, 3, 4, 5])

        self.sub_mean = common.MeanShift3D(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift3D(1.0, n_channels=n_channels, sign=1)

    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''

        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

        return flows_backward, flows_forward


    def get_flow_2frames(self, x):
        '''Get flow between frames t and t+1 from x.'''


        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        #print("x_1 before  x[:, :-1, :, :, :].shape : ", x[:, :-1, :, :, :].shape)
        #print("x_1 .shape ", x_1.shape)
        #print("x_2.shape :", x_2.shape)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                          zip(flows_backward, range(4))]

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                         zip(flows_forward, range(4))]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        '''Get flow between t and t+2 from (t,t+1) and (t+1,t+2).'''

        # backward
        d = flows_forward[0].shape[1]
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
            flows_backward2.append(torch.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
            flows_forward2.append(torch.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        '''Get flow between t and t+3 from (t,t+2) and (t+2,t+3).'''

        # backward
        d = flows_forward2[0].shape[1]
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
            flows_backward3.append(torch.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
            flows_forward3.append(torch.stack(flow_list, 1))

        return flows_backward3, flows_forward3


    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
        '''Parallel feature warping for 2 frames.'''

        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        # print('aligned x_backward.shape:', x_backward[0].shape)
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            # print('i:', i)
            # print('x_i.shape:', x_i.shape)
            # print('flow.shape:', flow.shape)
            warp = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')
            # print('warp.shape:', warp.shape)
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i+1 aligned towards i

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i-1 aligned towards i

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]



    def forward(self, x):
        #print('x.in.shape:', x.shape)
        res = x
        x_in = x
        
        xflow = x.clone().transpose(1, 2)
        
        # calculate flows
        flows_backward, flows_forward = self.get_flows(xflow)
        
        # warp input
        x_backward, x_forward = self.get_aligned_image_2frames(xflow,  flows_backward[0], flows_forward[0])
        
        x = torch.cat([xflow, x_backward, x_forward], 2)
        x = x.transpose(1, 2)
        # print('++++')
        #print("x.shape :", x.shape)
        x1 = self.conv_first(x)
        
        out = x1 + res
        out = self.add_mean(out)
        #print("out.shape ", out.shape)
        bs, c, d, h, w = out.shape
        out = out.view(bs, (d)*c, h, w)
        
        return out
