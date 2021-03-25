import torch
import pyro

from pyro.infer import Trace_ELBO
from pyro.util import warn_if_nan
from pyro.infer.util import torch_backward, torch_item

class TraceCCVAE_ELBO(Trace_ELBO):
    """
    Custom loss
    """
    def __init__(self, vae, *args, **kwargs):
        super(TraceCCVAE_ELBO, self).__init__(*args, **kwargs)
        self.vae = vae

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = self._loss(model, guide, args, kwargs)
        torch_backward(loss, retain_graph=self.retain_graph)
        loss = torch_item(loss)
        warn_if_nan(loss, "loss")
        return loss

    def _loss(self, model, guide, args, kwargs):

        loss = 0.0
        k = 100
        lqyx = self.vae.classifier_loss(*args, **kwargs, k=k)

        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):

            log_vi = model_trace.nodes['recon']["log_prob"]
            log_vi += model_trace.nodes['z_class']["log_prob"]
            log_vi += model_trace.nodes['z_style']["log_prob"]
            log_vi -= guide_trace.nodes['z_class']["log_prob"]
            log_vi -= guide_trace.nodes['z_style']["log_prob"]
            log_vi -= guide_trace.nodes['y']["log_prob"]
            lpy = model_trace.nodes['y']["log_prob"]

            w = torch.exp(guide_trace.nodes['y']["log_prob"] - lqyx) 

            elbo = (w * log_vi + lpy).sum()

            loss -= elbo

        loss /= self.num_particles

        loss -= lqyx.sum()
      
        return loss

