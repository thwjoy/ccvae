import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import pyro
import pyro.distributions as dist
import os
from .networks import (CondPrior, Classifier,
                       CELEBADecoder, CELEBAEncoder)

class SSVAE_CCVAE(torch.nn.Module):
    """
    Class that deals with the proposed M3 model
    """
    def __init__(self, num_classes, im_shape, z_dim,
                 prior_fn, use_cuda, class_name_fn,
                 load_data=None):
        super(SSVAE_CCVAE, self).__init__()
        self.class_name_fn = class_name_fn
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.z_classify = num_classes
        self.z_style = z_dim - self.z_classify
        self.im_shape = im_shape
        self.use_cuda = use_cuda
        self.im_dim = np.prod(im_shape)
        self._z_prior_fn = dist.Normal
        self.prior = prior_fn()
        
        self.encoder = CELEBAEncoder(z_dim=self.z_dim)
        self.decoder = CELEBADecoder(z_dim=self.z_dim)
        self.classifier = Classifier(dim=self.num_classes)
        self.cond_prior = CondPrior(dim=self.num_classes)
  
        if self.use_cuda:
            self.cuda()
            self.prior = self.prior.cuda()

    def model(self, xs, ys=None):
        pyro.module("ss_vae", self)
        bs = xs.shape[0]
        with pyro.plate("data", bs):
            ys = pyro.sample("y",
                             self._y_prior_fn(self.prior.expand(bs, -1)),
                             obs=ys)

            z_class = pyro.sample("z_class", self.cond_prior_fn(ys))
            z_style = pyro.sample("z_style", self._z_prior_fn(
                        *self._z_prior_params((bs, self.z_style))).to_event(1))
            z = torch.cat([z_class, z_style], dim=1)
            
            img = self.decoder(z).view(bs, -1)
            
            pyro.sample("recon", self.likelihood(img), obs=xs.reshape(bs, -1))

        return img

    def guide(self, xs, ys=None):
        with pyro.plate("data", xs.shape[0]):
            loc, scale = self.encoder(xs)
            locs_c, locs_s = loc.split([self.z_classify, self.z_style], 1)
            scales_c, scales_s = scale.split([self.z_classify, self.z_style], 1)
            z_class = pyro.sample("z_class", self._z_prior_fn(locs_c, scales_c).to_event(1))
            z_style = pyro.sample("z_style", self._z_prior_fn(locs_s, scales_s).to_event(1))
            pyro.sample("y", self._y_prior_fn(self.classifier(z_class.detach())), obs=ys) # detach to reduce variance of gradients

    def _y_prior_fn(self, alpha):
        return dist.Bernoulli(torch.sigmoid(alpha)).to_event(1)

    def _z_prior_params(self, shape):
        ones = torch.ones(shape)
        zeros = torch.zeros(shape)
        if self.use_cuda:
            ones, zeros = ones.cuda(), zeros.cuda()
        return zeros, ones

    def likelihood(self, img):
        return dist.Laplace(img, torch.ones_like(img)).to_event(1)

    def cond_prior_fn(self, y):
        z_loc_y, z_scale_y = self.cond_prior(y)
        return self._z_prior_fn(z_loc_y, z_scale_y).to_event(1)
        
    def classify(self, xs, ys=None):
        locs, scales = self.encoder(xs)
        locs, _ = locs.split([self.z_classify, self.z_style], 1)
        logits = self.classifier(locs)
        preds = torch.round(torch.sigmoid(logits))
        acc = None
        if ys is not None:
            acc = (preds.eq(ys)).float().mean()
        return preds, acc

    def accuracy(self, data_loader, cuda=False):
        acc = 0.0
        for (xs, ys) in data_loader:
            if cuda:
                xs, ys = xs.cuda(), ys.cuda()
            _, batch_acc = self.classify(xs, ys)
            acc += batch_acc
        return acc / len(data_loader)

    def classifier_loss(self, xs, ys, k=100):
        locs, scales = self.encoder(xs)
        zs = self._z_prior_fn(*self.encoder(xs)).rsample(torch.tensor([k]))
        zs, _ = zs.split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zs.view(-1, self.z_classify))
        d = self._y_prior_fn(logits)
        ys = ys.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(ys).view(k, xs.shape[0])
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

    def reconstruct_img(self, x):
        loc, scale = self.encoder(x)
        z = self._z_prior_fn(loc, scale).sample()
        return self.decoder(z)

    def latent_walk(self, image, save_dir):
        mult = 5
        num_imgs = 5
        z_ = self._z_prior_fn(*self.encoder(image.unsqueeze(0))).sample()
        for i in range(self.num_classes):
            y_1 = torch.zeros(1, self.num_classes)
            if self.use_cuda:
                y_1 = y_1.cuda()
            locs_false, scales_false = self.cond_prior(y_1)
            y_1[:, i].fill_(1.0)
            locs_true, scales_true = self.cond_prior(y_1)
            sign = torch.sign(locs_true[:, i] - locs_false[:, i])
            # y axis
            z_1_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
            z_1_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()   
            for j in range(self.num_classes):
                z = z_.clone()
                z = z.expand(num_imgs**2, -1).contiguous()
                if i == j:
                    continue
                y_2 = torch.zeros(1, self.num_classes)
                if self.use_cuda:
                    y_2 = y_2.cuda()
                locs_false, scales_false = self.cond_prior(y_2)
                y_2[:, i].fill_(1.0)
                locs_true, scales_true = self.cond_prior(y_2)
                sign = torch.sign(locs_true[:, i] - locs_false[:, i])
                # x axis
                z_2_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
                z_2_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()

                # construct grid
                range_1 = torch.linspace(z_1_false_lim, z_1_true_lim, num_imgs)
                range_2 = torch.linspace(z_2_false_lim, z_2_true_lim, num_imgs)
                grid_1, grid_2 = torch.meshgrid(range_1, range_2)
                z[:, i] = grid_1.reshape(-1)
                z[:, j] = grid_2.reshape(-1)

                imgs = self.decoder(z).view(-1, *self.im_shape)
                grid = make_grid(imgs, nrow=num_imgs)
                save_image(grid, os.path.join(save_dir, "latent_walk_%s_and_%s.png"
                                              % (self.class_name_fn(i), self.class_name_fn(j))))

        mult = 8
        for j in range(self.num_classes):
            z = z_.clone()
            z = z.expand(10, -1).contiguous()
            y = torch.zeros(1, self.num_classes)
            if self.use_cuda:
                y = y.cuda()
            locs_false, scales_false = self.cond_prior(y)
            y[:, i].fill_(1.0)
            locs_true, scales_true = self.cond_prior(y)
            sign = torch.sign(locs_true[:, i] - locs_false[:, i])
            z_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
            z_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()
            range_ = torch.linspace(z_false_lim, z_true_lim, 10)
            z[:, j] = range_

            imgs = self.decoder(z).view(-1, *self.im_shape)
            grid = make_grid(imgs, nrow=10)
            save_image(grid, os.path.join(save_dir, "latent_walk_%s.png"
                                              % self.class_name_fn(j)))

    def save_models(self, path):
        torch.save(self.encoder, os.path.join(path,'encoder.pt'))
        torch.save(self.decoder, os.path.join(path,'decoder.pt'))
        torch.save(self.classifier, os.path.join(path,'classifier.pt'))
        torch.save(self.cond_prior, os.path.join(path,'cond_prior.pt'))

