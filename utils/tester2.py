import os
import time
import imageio

import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np

"""
This function make output from patches of image.
To do so, opt.test_patches shoule be set True.
If opt.test_patches is False, we test the network with shole image. --test_image -> set opt.test_patches False

We test divided patches because waveletdl cannot make the proper output when the input is whole image,
thus we divide the image into the patch with trained patch size
To do this, we need to pad the image, divide into patches, run the network, reconstruct the output image from patches,
and unpad the output image.

If SWT is not used, the whole image can be used.
"""

def test_net_by_tensor_patches(opt, model, tensor_x):
    mp_start_time = time.time()

    img_patch_dataset = Tensor2PatchDataset(opt, tensor_x)
    img_patch_dataloader = data.DataLoader(dataset=img_patch_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=opt.n_threads)

    img_shape = img_patch_dataset.get_img_shape()
    pad_img_shape = img_patch_dataset.get_padded_img_shape()

    mp_end_time = time.time()
    if not opt.is_train: print("[*] Making patches: {:4f}s".format((mp_end_time - mp_start_time)))

    net_start_time = time.time()
    if not opt.is_train: print("[*] Network started")
    out_list = []

    for batch in img_patch_dataloader:
        # print("batch:", batch.shape)
        input = {
            'x': batch,
            'target': None
        }
        with torch.no_grad():
            model.set_input(input)
            model.test()

        out = model.out
        # out = out.to('cpu').detach().numpy()
        out_list.append(out)
        
    net_end_time = time.time()
    if not opt.is_train: print("[*] Network process: {:4f}s".format((net_end_time - net_start_time)))

    out = torch.cat(out_list, axis=0)
    # print("out.shape:", out.shape)
    
    recon_start_time = time.time()
    
    out_img = recon_tensor_arr_patches(out, pad_img_shape[3], pad_img_shape[2], opt.patch_size, opt.patch_offset)
    out_img = unpad_tensor(out_img, opt.patch_offset, img_shape)
    
    recon_end_time = time.time()
    if not opt.is_train: print("[*] Reconstruction time: {:4f}s".format((recon_end_time - recon_start_time)))
    if not opt.is_train: print("[*] Total time {:.4f}s".format(recon_end_time - mp_start_time))

    return out_img


def pad_tensor(tensor_img, patch_size, patch_offset):
    # print("tensor_img.shape:", tensor_img.shape)
    stride = patch_size - 2 * patch_offset
    bs, c, h, w = tensor_img.shape

    rw = (patch_offset + w) % stride
    rh = (patch_offset + h) % stride
    w_stride_pad_size = stride - rw
    h_stride_pad_size = stride - rh

    stride_pad_w = patch_offset + w + w_stride_pad_size
    stride_pad_h = patch_offset + h + h_stride_pad_size

    w_pad_size = w_stride_pad_size + patch_size
    h_pad_size = h_stride_pad_size + patch_size

    npad = (patch_offset, h_pad_size, patch_offset, w_pad_size)

    tensor_img = F.pad(tensor_img, npad, mode='reflect')
    # print("padded tensor_img.shape:", tensor_img.shape)

    return tensor_img

def unpad_tensor(tensor_img, patch_offset, tensor_img_shape):
    bs, c, h, w = tensor_img_shape
    tensor_ret = tensor_img[:, :, patch_offset:patch_offset+h, patch_offset:patch_offset+w]
    return tensor_ret

def make_tensor_arr_patches(tensor_img, patch_size, patch_offset):
    bs, c, h, w = tensor_img.shape

    assert bs == 1

    stride = patch_size - 2 * patch_offset
    mod_h = h - np.mod(h - patch_size, stride)
    mod_w = w - np.mod(w - patch_size, stride)
    
    num_patches = (mod_h // stride) * (mod_w // stride)

    patch_arr = torch.zeros((num_patches, c, patch_size, patch_size), dtype=tensor_img.dtype)

    ps = patch_size

    patch_idx = 0
    for y in range(0, mod_h - stride + 1, stride):
        for x in range(0, mod_w - stride + 1, stride):
            patch = tensor_img[:, :, y:y+ps, x:x+ps]
            patch_arr[patch_idx] = patch
            patch_idx += 1

    return patch_arr

def recon_tensor_arr_patches(patch_arr, width, height, patch_size, patch_offset):
    stride = patch_size - 2 * patch_offset

    _, c, _, _ = patch_arr.shape

    tesnor_img = torch.zeros((1, c, height, width), dtype=patch_arr.dtype)

    mod_h = height - np.mod(height - 2 * patch_offset, stride)
    mod_w = width - np.mod(width - 2 * patch_offset, stride)

    ps = patch_size
    po = patch_offset

    patch_idx = 0
    for y in range(0, mod_h - (patch_size - patch_offset) + 1, stride):
        for x in range(0, mod_w - (patch_size - patch_offset) + 1, stride):
            patch = patch_arr[patch_idx]
                
            tesnor_img[:, : ,y+po:y+ps-po, x+po:x+ps-po] = patch[:, po:-po, po:-po]
            patch_idx += 1

    return tesnor_img


"""
This Dataset class converts one tensor image into divided patches
"""
class Tensor2PatchDataset(data.Dataset):
    def __init__(self, opt, tensor_img):
        super(Tensor2PatchDataset, self).__init__()
        self.tensor_img_shape = tensor_img.shape
        self.opt = opt

        padded_tensor = pad_tensor(tensor_img, opt.patch_size, opt.patch_offset)
        self.padded_tensor_shape =padded_tensor.shape
        self.tensor_patches = make_tensor_arr_patches(padded_tensor, opt.patch_size, opt.patch_offset)

    def __getitem__(self, idx):
        patch = self.tensor_patches[idx]
        return patch

    def __len__(self):
        return len(self.tensor_patches)

    def get_img_shape(self):
        return self.tensor_img_shape

    def get_padded_img_shape(self):
        return self.padded_tensor_shape



"""
Some functions for save results
Is it better to create class object rather than functions?
"""

def calc_metrics(tensors_dict):
    x = tensors_dict['x']
    x = x[:, x.size(1)//2]
    out = tensors_dict['out']
    target = tensors_dict['target']
    target = target[:, target.size(1)//2]

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(out, x)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)

    return noise_loss, noise_psnr, out_loss, out_psnr

def save_tensors(opt, data_idx, tensors_dict):
    tensor_x = tensors_dict['x']
    tensor_out = tensors_dict['out']
    tensor_target = tensors_dict['target']
    filenames = tensors_dict['filename']

    _, n, _, _ = tensor_target.shape
    assert n == opt.n_inputs

    x = tensor_x.detach().to('cpu').numpy()
    x_mean = tensor_x.mean(1)
    x_mean = x_mean.detach().to('cpu').numpy()
    x_mean = x_mean.transpose((1, 2, 0)).squeeze()

    test_results_dir= os.path.join(opt.test_results_dir, opt.test_datasets[data_idx])
    os.makedirs(test_results_dir, exist_ok=True)
    compare_test_results_dir = os.path.join(test_results_dir, 'compare')
    os.makedirs(compare_test_results_dir, exist_ok=True)

    # x = x.detach().to('cpu').numpy()
    out = tensor_out.detach().to('cpu').numpy().squeeze()
    target = tensor_target.detach().to('cpu').numpy()
    # print('x.shape:', x.shape)
    # print('target.shape:', target.shape)
    targetd = target[:, n//2].squeeze()
    
    # x = x.squeeze().transpose((1, 2, 0))
    # out = out.transpose((1, 2, 0)).squeeze()
    # targetd = target.squeeze().transpose((1, 2, 0))
        
    out[out > 1.0] = 1.0
    out[out < 0.0] = 0.

    # print('x0.shape:', x0.shape)
    # print('x_mean.shape:', x_mean.shape)
    x_concat = []
    for i in range(x.shape[1]):
        # print('x[:, {}].shape: {}'.format(i, x[:, i].shape))
        x_concat.append(x[:, i].squeeze())
    # print('x_mean.shape:', x_mean.shape)
    # print('out.shape:', out.shape)
    # print('targetd.shape:', targetd.shape)
    compare_img = np.concatenate((*x_concat, x_mean, out, targetd), axis=1)

    fmt = '.tiff'
    out_fn_path = os.path.join(test_results_dir, 'out-' + filenames + fmt)
    compare_fn_path = os.path.join(compare_test_results_dir, filenames + fmt)

    print("Writing {}".format(os.path.abspath(out_fn_path)))
    imageio.imwrite(out_fn_path, out)
    imageio.imwrite(compare_fn_path, compare_img)

def save_metrics(opt, data_idx, fi, filename, noise_loss, noise_psnr, out_loss, out_psnr):
    report_path = os.path.join(opt.test_results_dir, 'report.csv')

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("dataset,idx,filename,noise_loss,noise_psnr,loss,psnr\n")

    with open(report_path, 'a') as f:
        f.write("{},{},{},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            opt.test_datasets[data_idx], fi, filename, noise_loss, noise_psnr, out_loss, out_psnr
        ))

def save_results(opt, data_idx, fi, filename, tensors_dict):
    tensor_x = tensors_dict['x']
    tensor_out = tensors_dict['out']
    tensor_target = tensors_dict['target']
    case = tensors_dict['case']
    filenames = tensors_dict['filename']

    _, n, _, _ = tensor_target.shape
    assert n == opt.n_inputs

    # x_all = tensor_x
    x_in = torch.cat((tensor_x[:, :n//2], tensor_x[:, n//2:]), dim=1)
    x_mean = x_in.mean(1).detach().to('cpu').squeeze()
    # print('x_mean.shape:', x_mean.shape)
    
    x_mid = tensor_x[:, n//2].detach().to('cpu').squeeze()
    out = tensors_dict['out'].detach().to('cpu').squeeze()
    target = tensors_dict['target'].detach().to('cpu')
    target = target[:, n//2].squeeze()

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(x_mid, target)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    mean_loss = mse_criterion(x_mean, target)
    mean_psnr = 10 * torch.log10(1 / mean_loss)

    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)

    np_x_all = tensor_x.detach().to('cpu').numpy()
    # np_x_all = np_x_all.transpose((1, 2, 0)).squeeze()

    np_x_mean = x_mean.detach().to('cpu').numpy()
    # np_x_mean = np_x_mean.transpose((1, 2, 0)).squeeze()
    np_out = tensor_out.detach().to('cpu').numpy().squeeze()
    np_target = tensor_target.detach().to('cpu').numpy()
    # print('x.shape:', x.shape)
    # print('target.shape:', target.shape)
    np_targetd = np_target[:, n//2].squeeze()

    x_concat = []
    for i in range(n):
        # print('x[:, {}].shape: {}'.format(i, x[:, i].shape))
        x_concat.append(np_x_all[:, i].squeeze())

    compare_img = np.concatenate((*x_concat, np_x_mean, np_out, np_targetd), axis=1)

    test_results_dir= os.path.join(opt.test_results_dir, opt.test_datasets[data_idx])
    test_results_case_dir = os.path.join(test_results_dir, case)
    os.makedirs(test_results_case_dir, exist_ok=True)
    compare_test_results_dir = os.path.join(test_results_dir, 'compare')
    compare_test_results_case_dir = os.path.join(compare_test_results_dir, case)
    os.makedirs(compare_test_results_case_dir, exist_ok=True)


    fmt = '.tiff'
    out_fn_path = os.path.join(test_results_case_dir, 'out-' + filenames + fmt)
    compare_fn_path = os.path.join(compare_test_results_case_dir, filenames + fmt)

    print("Writing {}".format(os.path.abspath(out_fn_path)))
    imageio.imwrite(out_fn_path, out)
    imageio.imwrite(compare_fn_path, compare_img)

    report_path = os.path.join(opt.test_results_dir, 'report.csv')

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("dataset,idx,filename,noise_loss,noise_psnr,mean_loss,mean_psnr,loss,psnr\n")

    with open(report_path, 'a') as f:
        f.write("{},{},{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            opt.test_datasets[data_idx], fi, filename, noise_loss, noise_psnr, mean_loss,mean_psnr,out_loss, out_psnr
        ))

    return noise_loss, noise_psnr, mean_loss, mean_psnr, out_loss, out_psnr

def save_summary(opt, data_idx, avg_noise_loss, avg_noise_psnr, mean_avg_loss, mean_avg_psnr, avg_loss, avg_psnr):
    report_path = os.path.join(opt.test_results_dir, 'summary.txt')

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("*** SUMMARY ***\n")

    with open(report_path, 'a') as f:
        f.write("{} - Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Mean Loss: {:.8f}, Average Mean PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}\n".format(
            opt.test_datasets[data_idx], avg_noise_loss, avg_noise_psnr, mean_avg_loss, mean_avg_psnr, avg_loss, avg_psnr
        ))