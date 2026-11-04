import torch, os, time, math, warnings, numpy as np
import torchvision.transforms as tfs
from torch import nn, optim
from torch.backends import cudnn
from torchvision.models import vgg16
from models import *
from metrics import psnr, ssim
from option import opt, model_name, log_dir
from data_utils import loader_train, loader_test  # Automatically selected Dataloader

warnings.filterwarnings('ignore')

print('log_dir :', log_dir)
print('model_name:', model_name)

# ==========================
# Define model
# ==========================
models_ = {
    'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
}

start_time = time.time()
T = opt.steps

# ==========================
# Cosine LR scheduler
# ==========================
def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    return 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr

# ==========================
# check the results with PSNR, SSIM
# ==========================
def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims, psnrs = [], []
    for inputs, targets in loader_test:
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = net(inputs)
        ssims.append(ssim(pred, targets).item())
        psnrs.append(psnr(pred, targets))
    return np.mean(ssims), np.mean(psnrs)

# ==========================
# Training
# ==========================
def train(net, loader_train, loader_test, optimizer, criterion):
    losses, ssims, psnrs = [], [], []
    start_step = 0
    best_ssim, best_psnr = 0, 0

    # Check resume, whether the original file exists
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'Resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir, map_location=opt.device, weights_only=False)

        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        losses = ckp.get('losses', [])
        ssims = ckp.get('ssims', [])
        psnrs = ckp.get('psnrs', [])
        best_ssim = ckp.get('best_ssim', 0)
        best_psnr = ckp.get('best_psnr', 0)
        print(f'Start step: {start_step} --- continue training')
    else:
        print('Train from scratch ***')

    train_iter = iter(loader_train)

    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        lr = lr_schedule_cosdecay(step, T) if not opt.no_lr_sche else opt.lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loader_train)
            x, y = next(train_iter)

        x, y = x.to(opt.device), y.to(opt.device)
        out = net(x)
        loss = criterion[0](out, y)
        if opt.perloss:
            loss2 = criterion[1](out, y)
            loss += 0.04 * loss2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        print(f'\rtrain loss : {loss.item():.5f} | step : {step}/{opt.steps} | lr :{lr:.7f} | time_used :{(time.time()-start_time)/60:.1f} min',
              end='', flush=True)

        # evaluate the result with the previous one
        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)
            print(f'\nStep : {step} | SSIM: {ssim_eval:.4f} | PSNR: {psnr_eval:.4f}')
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            if ssim_eval > best_ssim and psnr_eval > best_psnr:
                best_ssim, best_psnr = ssim_eval, psnr_eval
                save_path = f"{opt.model_dir.replace('.pk','')}-{step}.pk"
                torch.save({
                    'step': step,
                    'best_ssim': best_ssim,
                    'best_psnr': best_psnr,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict()
                }, save_path)
                print(f"✅ Model saved at step {step} | path: {save_path}")

    os.makedirs('./numpy_files', exist_ok=True)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


# ==========================
# execute the main
# ==========================
if __name__ == "__main__":
    net = models_[opt.net].to(opt.device)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = [nn.L1Loss().to(opt.device)]
    if opt.perloss:
        vgg_model = vgg16(pretrained=True).features[:16].to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(opt.device))

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()),
                           lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    train(net, loader_train, loader_test, optimizer, criterion)
      