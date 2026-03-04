import time
import torch
import random
from utils.tester import test_net_by_tensor_patches


def train_net(opt, model, train_dataloader):
    print("*** Training phase ***")
    model.train()
    avg_loss = 0.0
    avg_psnr = 0.0

    start_time = time.time()
    for i, batch in enumerate(train_dataloader, 1):
        #print("herea", len(batch))
        x, target = batch[0], batch[1]
        #print('loader x.shape:', x.shape)

        input = {
            'x': x,
            'target': target
        }
        model.set_input(input)
        model.optimize_parameters()

        end_time = time.time()
        model.log_loss(opt, "Training", end_time - start_time, i, len(train_dataloader))

        batch_loss, batch_psnr = model.get_batch_loss_psnr()
        avg_loss += batch_loss
        avg_psnr += batch_psnr

    avg_loss, avg_psnr = avg_loss / i, avg_psnr / i
    # log_epoch_loss("Training", avg_loss, avg_psnr)
    print("===> Training avg_loss: {:.8f}, avg_psnr: {:.8f}".format(avg_loss, avg_psnr))
    return avg_loss, avg_psnr

def valid_net(opt, model, test_dataloader):
    model.eval()

    start_time = time.time()
    itr = 0
    avg_loss = 0.0
    avg_psnr = 0.0

    for i, batch in enumerate(test_dataloader, 1):
        x, target = batch[0], batch[1]
        # print('loader x.shape:', x.shape)

        with torch.no_grad():
            if opt.test_random_patch or not opt.test_patches:
                input = {
                    'x': x,
                    'target': target
                }
                model.set_input(input)
                model.test()

                # out and target should be detach()
                out = model.out.detach()
                target = model.target.detach()
            else:
                out = test_net_by_tensor_patches(opt, model, x)

        # Make sure every tensor variable is in the same device and detached
        # all tensor variables x, out, and target should be detach()
        # if not, there would memory leak
        x = x.to(opt.device).detach()
        out = out.to(opt.device).detach()
        target = target.to(opt.device).detach()

        batch_loss, batch_psnr = calc_loss_psnr(out, target)
        
        avg_loss += batch_loss
        avg_psnr += batch_psnr
        itr += 1

        end_time = time.time()
        print("Validation {:.3f}s => Epoch[{}/{}]({}/{}): Batch Loss: {:.8f}, PSNR: {:.8f}".format(
            end_time - start_time, opt.epoch, opt.n_epochs, i, len(test_dataloader), batch_loss.item(), batch_psnr.item())
        )

    avg_loss, avg_psnr = avg_loss / itr, avg_psnr / itr
    print("===> Validation - Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        avg_loss, avg_psnr
    ))
    return avg_loss, avg_psnr

def calc_loss_psnr(out, target):
    mse_criterion = torch.nn.MSELoss()
    mse_loss = mse_criterion(out, target)
    psnr = 10 * torch.log10(1 / mse_loss)
    return mse_loss, psnr


# def save_random_tensors(opt, epoch, model, test_dataloaders):
#     exprdir = opt.exprdir
    
#     for di, test_dataloader in enumerate(test_dataloaders):
#         random_idx = random.sample(test_dataloader, opt.n_saves)
        
#         for i, batch in enumerate(test_dataloader, 1):
#             x, target, filename = batch[0], batch[1], batch[2]
#             input = {
#                 'x': x,
#                 'target': target
#             }
#             model.set_input(input)
#             model.test()

#             # out and target should be detach()
#             out = model.out.detach()
#             target = model.target.detach()
#         else:
#             out = test_net_by_tensor_patches(opt, model, x)

#             test_results_dir= os.path.join(opt.exprdir, opt.test_datasets[di])
#             os.makedirs(test_results_dir, exist_ok=True)
#             compare_test_results_dir = os.path.join(test_results_dir, 'compare')
#             os.makedirs(compare_test_results_dir, exist_ok=True)

#             x = x[0].detach().to('cpu').numpy()
#             out = out[0].detach().to('cpu').numpy()
#             target = target[0].detach().to('cpu').numpy()

#             # print("x.shape:", x.shape)
#             x = x.transpose((1, 2, 0)).squeeze()
#             out = out.transpose((1, 2, 0)).squeeze()
#             target = target.transpose((1, 2, 0)).squeeze()

#             out[out > 1.0] = 1.0
#             out[out < 0.0] = 0.
#             if c == 3:
#                 x = x * 255
#                 out = out * 255
#                 target = target * 255
#                 x = np.rint(x)
#                 out = np.rint(out)
#                 target = np.rint(target)
#                 x  = x.astype(np.uint8)
#                 out  = out.astype(np.uint8)
#                 target  = target.astype(np.uint8)

#             compare_img = np.concatenate((x, out, target), axis=1)

#             fmt = '.tiff' if c == 1 else '.png'
#             out_fn_path = os.path.join(test_results_dir, 'out-' + str(epoch) + filenames[0] + fmt)
#             compare_fn_path = os.path.join(compare_test_results_dir, str(epoch) + filenames[0] + fmt)

#             print("Writing {}".format(os.path.abspath(out_fn_path)))
#             imageio.imwrite(out_fn_path, out)
#             imageio.imwrite(compare_fn_path, compare_img)