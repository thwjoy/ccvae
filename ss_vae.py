import argparse
import torch
import torchvision.utils 
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from utils.custom_loss import TraceCCVAE_ELBO
from utils.dataset_cached import setup_data_loaders, CELEBACached, CELEBA_EASY_LABELS
from models.semisup_vae import SSVAE_CCVAE
import numpy as np
import os
import random


def main(args):


    seed = 1234
    pyro.set_rng_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pyro.clear_param_store()

    data_loaders = setup_data_loaders(CELEBACached,
                                      args.batch_size,
                                      sup_frac=args.sup_frac,
                                      root='./data/datasets/celeba')

    ss_vae = SSVAE_CCVAE(use_cuda=args.cuda,
                         num_classes=len(CELEBA_EASY_LABELS),
                         im_shape=CELEBACached.shape,
                         z_dim=args.z_dim,
                         prior_fn=CELEBACached.prior_fn,
                         class_name_fn=lambda i : str(CELEBA_EASY_LABELS[i]))

    adam_params = {"lr": args.learning_rate, "betas": (0.9, 0.999)}
    optim = Adam(adam_params)

    loss_sup = SVI(ss_vae.model, ss_vae.guide, optim, loss=TraceCCVAE_ELBO(ss_vae))
    loss_unsup = SVI(ss_vae.model, ss_vae.guide, optim, loss=Trace_ELBO())
        
    for epoch in range(0, args.num_epochs):

        if args.sup_frac == 1.0:
            batches_per_epoch = len(data_loaders["sup"])
            period_sup_batches = 1
            sup_batches = batches_per_epoch
        elif args.sup_frac > 0.0: 
            sup_batches = len(data_loaders["sup"])
            unsup_batches = len(data_loaders["unsup"])
            batches_per_epoch = sup_batches + unsup_batches
            period_sup_batches = int(batches_per_epoch / sup_batches)
        elif args.sup_frac == 0.0:
            sup_batches = 0.0
            batches_per_epoch = len(data_loaders["unsup"])
            period_sup_batches = np.Inf
        else:
            assert False, "Data frac not correct"

        epoch_losses_sup = 0.0
        epoch_losses_unsup = 0.0

        if args.sup_frac != 0.0:
            sup_iter = iter(data_loaders["sup"])
        if args.sup_frac != 1.0:
            unsup_iter = iter(data_loaders["unsup"])

        ctr_sup = 0

        for i in range(batches_per_epoch):
            is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches

            if is_supervised:
                (xs, ys) = next(sup_iter)
                ctr_sup += 1
            else:
                (xs, ys) = next(unsup_iter)

            if args.cuda:
                xs, ys = xs.cuda(), ys.cuda()

            if is_supervised:
                epoch_losses_sup += loss_sup.step(xs=xs, ys=ys)
            else:
                epoch_losses_unsup += loss_unsup.step(xs=xs)

        if args.sup_frac != 0.0:        
            with torch.no_grad():
                validation_accuracy = ss_vae.accuracy(data_loaders['valid'], args.cuda)
        else:
            validation_accuracy = np.nan
        
        print("[Epoch %03d] Sup Loss %.3f, Unsup Loss %.3f, Val Acc %.3f" % 
                (epoch, epoch_losses_sup, epoch_losses_unsup, validation_accuracy))
    ss_vae.save_models(args.data_dir)
    return test(args, ss_vae, data_loaders)


def test(args, ss_vae, data_loaders):

    im_shape = CELEBACached.shape

    test_accuracy = ss_vae.accuracy(data_loaders['test'], args.cuda)
    xs = data_loaders['test'].dataset.fixed_imgs
    xs = xs.cuda() if args.cuda else xs
    ss_vae.latent_walk(xs[5], args.data_dir)
    with open(os.path.join(args.data_dir, 'results.txt'), 'w') as fp:
        fp.write('Test acc %.3f\n' % test_accuracy)
    
    imgs = ss_vae.reconstruct_img(xs[:25]).view(-1, *im_shape)
    torchvision.utils.save_image(torchvision.utils.make_grid(imgs, nrow=5),
                                os.path.join(args.data_dir, 'recons.png'))
    torchvision.utils.save_image(torchvision.utils.make_grid(xs[:25].view(-1, *im_shape), nrow=5),
                                os.path.join(args.data_dir, 'origs.png'))
    
    return test_accuracy

def parser_args(parser):
    parser.add_argument('--cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=95, type=int,
                        help="number of epochs to run")
    parser.add_argument('-sup', '--sup-frac', default=1.0,
                        type=float, help="supervised fractional amount of the data i.e. "
                                         "how many of the images have supervised labels."
                                         "Should be a multiple of train_size / batch_size")
    parser.add_argument('-zd', '--z_dim', type=int, default=45,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data path')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()

    args.run_identifier = 'f_' + str(args.sup_frac)
    args.data_dir = os.path.join(args.data_dir,
                                'vae_results',
                                args.run_identifier)
    os.makedirs(args.data_dir, exist_ok=True)

    main(args)

