"""
VAEs and variants of VAEs (for generic data)

Models for joint generative models over some attribute (discrete/binary)
and the continuous beat signal
"""
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pyprind
from drvae.model import base, train
import drvae.misc
from drvae.model.aes import ConvAEEncoder, ConvAEDecoder, LinearAEEncoder, LinearAEDecoder

ln2pi = np.log(2*np.pi)


######################
# VAE Model Classes  #
######################
class VAE(base.Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        ll_name = kwargs.get("loglike_function", "gaussian")
        if ll_name == "gaussian":
            self.ll_fun = recon_loglike_function
        elif ll_name == "bernoulli":
            self.ll_fun = binary_recon_loglike_function
        elif ll_name == "mse":
            self.ll_fun = mse_loss

        print(ll_name)
        print(self.ll_fun)

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x)
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def reparameterize(self, **kwargs):
        raise NotImplementedError

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        print(Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)
        self.fit_res = train.fit_vae(self, Xtrain, Xval, Xtest,
                                           Ytrain, Yval, Ytest, **kwargs)
        return self.fit_res

    def reconstruction_error(self, X):
        dset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.zeros(X.shape[0], 1))
        loader = torch.utils.data.DataLoader(dset, batch_size=512)
        errors = []
        for bi, (data, _) in enumerate(pyprind.prog_bar(loader)):
            data = Variable(data).cuda()
            rX = self.forward(data)[0]
            err = torch.abs(data - rX).mean(-1).mean(-1)
            errors.append(err.cpu())
        return torch.cat(errors)

    # andy's version
    def lossfun(self, data, recon_data, target, mu, logvar):
        # added contiguous() for running things on a CPU. 
        # https://github.com/agrimgupta92/sgan/issues/22
        # Note, currently target isn't used. Maybe kept for conditional VAE?
        # can be negative, https://stats.stackexchange.com/questions/319859/can-log-likelihood-funcion-be-positive
        recon_ll = self.ll_fun(
            recon_data.contiguous().view(recon_data.shape[0], -1),
            data.contiguous().view(data.shape[0], -1))
        kl_to_prior = kldiv_to_std_normal(mu, logvar)
        
        if self.ll_fun == mse_loss:
            return torch.mean(recon_ll + kl_to_prior)
        else:
            return -torch.mean(recon_ll - kl_to_prior)

class ConvVAE(VAE):
      def __init__(self, hparams, **kwargs):
          """
            Parameters
            ----------
            hparams : :obj:`dict`
                - 'model_type' (:obj:`int`): 'conv' | 'linear'
                - 'model_class' (:obj:`str`): 'ae' | 'vae'
                - 'y_pixels' (:obj:`int`)
                - 'x_pixels' (:obj:`int`)
                - 'n_input_channels' (:obj:`int`)
                - 'n_ae_latents' (:obj:`int`)
                - 'fit_sess_io_layers; (:obj:`bool`): fit session-specific input/output layers
                - 'ae_encoding_x_dim' (:obj:`list`)
                - 'ae_encoding_y_dim' (:obj:`list`)
                - 'ae_encoding_n_channels' (:obj:`list`)
                - 'ae_encoding_kernel_size' (:obj:`list`)
                - 'ae_encoding_stride_size' (:obj:`list`)
                - 'ae_encoding_x_padding' (:obj:`list`)
                - 'ae_encoding_y_padding' (:obj:`list`)
                - 'ae_encoding_layer_type' (:obj:`list`)
                - 'ae_decoding_x_dim' (:obj:`list`)
                - 'ae_decoding_y_dim' (:obj:`list`)
                - 'ae_decoding_n_channels' (:obj:`list`)
                - 'ae_decoding_kernel_size' (:obj:`list`)
                - 'ae_decoding_stride_size' (:obj:`list`)
                - 'ae_decoding_x_padding' (:obj:`list`)
                - 'ae_decoding_y_padding' (:obj:`list`)
                - 'ae_decoding_layer_type' (:obj:`list`)
                - 'ae_decoding_starting_dim' (:obj:`list`)
                - 'ae_decoding_last_FF_layer' (:obj:`bool`)
           """
          super(ConvVAE, self).__init__(**kwargs)
          self.hparams = hparams
          self.model_type = self.hparams['model_type']
          self.scale_pixels = kwargs.get("scale_pixels", False)
          self.img_size = (
                self.hparams['n_input_channels'],
                self.hparams['y_pixels'],
                self.hparams['x_pixels'])
          self.encoding_net = None
          self.decoding_net = None
          self.build_model()
          
      def __str__(self):
        """Pretty print the model architecture."""
        format_str = '\nAutoencoder architecture\n'
        format_str += '------------------------\n'
        format_str += self.encoding_net.__str__()
        format_str += self.decoding_net.__str__()
        format_str += '\n'
        return format_str

      def build_model(self):
        """Construct the model using hparams."""
        self.encoding_net = ConvAEEncoder(self.hparams)
        self.decoding_net = ConvAEDecoder(self.hparams)

      def reparameterize(self, mu, logvar):
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)

      def forward(self, x, dataset=None, use_mean=False):
        """Process input data.
        Parameters
        ----------
        x : :obj:`torch.Tensor` object
            input data
        dataset : :obj:`int`
            used with session-specific io layers
        Returns
        -------
        :obj:`tuple`
            - y (:obj:`torch.Tensor`): output of shape (n_frames, n_channels, y_pix, x_pix)
            - x (:obj:`torch.Tensor`): hidden representation of shape (n_frames, n_latents)
        To-Do: check how it'll work by changing x. connect with train script, and loss.
        Ideal - scaling occurs inside the forward method of the VAE,
        no need to worry about that outside of this module.'
        """
        if self.scale_pixels:
            #x = x/2048.0 + 0.5
            x = x/1024.0
        mu, lnvar, pool_idx, outsize = self.encoding_net(x, dataset=dataset)
            # now we reparametrize and push through decoder
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, lnvar) # sample using reparam trick
        x_bar = self.decoding_net(z , pool_idx, outsize, dataset=dataset)
        if self.scale_pixels:
            #x_bar = (x_bar - 0.5)*2048.0
            x_bar = x_bar*1024.0

        return x_bar, z, mu, lnvar
    
class ConvDRVAE(ConvVAE):
    """ adds a discriminative model and an associated penalty """
    def set_discrim_model(self, discrim_model, 
                          discrim_beta,
                          dim_out_to_use):
        # # assert that there are 0 trainalbe params in discrim model
        # assert(len(list(filter(lambda p: p.requires_grad, 
        #                        discrim_model.parameters())))==0)
        self.discrim_model = [discrim_model]
        self.discrim_beta = discrim_beta
        self.dim_out_to_use = dim_out_to_use # chose dimension of the discrim output

    def lossfun(self, data, recon_data, 
                target, mu, logvar, 
                scale_down_image_loss):
        
        # vae ELBO loss
        if scale_down_image_loss:
            # currently pixels in [-1024, 1024], 
            # don't want it to dominate the equation
            vae_loss = super(ConvDRVAE, self).lossfun(
            data/1024.0, recon_data/1024.0, target, mu, logvar)
        else:
            vae_loss = super(ConvDRVAE, self).lossfun(
            data, recon_data, target, mu, logvar)

        if self.discrim_beta == 0:
            return vae_loss

        # push data and recond through discrim_model
        # ToDo - validate that this works.
        zdiscrim_data  = self.discrim_model[0](data)
        zdiscrim_recon = self.discrim_model[0](recon_data)
        # squared error (ToDo: consider implementing binary KL)
        disc_loss = self.discrim_beta * \
            torch.sum((zdiscrim_data.clone()[:, self.dim_out_to_use]
                       -zdiscrim_recon.clone()[:, self.dim_out_to_use])**2)

        assert ~np.isnan(vae_loss.clone().detach().cpu())
        assert ~np.isnan(disc_loss.clone().detach().cpu())
        return vae_loss + disc_loss
    


        
# class BeatConvVAE(VAE):
#     """ Convolutional VAE for a single Beat ... conv - deconv model
#     """
#     def __init__(self, **kwargs):
#         super(BeatConvVAE, self).__init__(**kwargs)
#         n_channels      = kwargs.get("n_channels")  # 3 for long leads
#         n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
#         self.latent_dim = kwargs.get("latent_dim")
#         self.data_shape = (n_channels, n_samples)
#         self.verbose    = kwargs.get("verbose", False)
#         kernel_size     = kwargs.get("kernel_size", 8)

#         self.encode_net = conv.BeatConvMLP(n_channels=n_channels,
#                                            n_samples=n_samples,
#                                            n_outputs=50)
#         self.erelu        = nn.ReLU()
#         self.encode_mu    = nn.Linear(50, self.latent_dim)
#         self.encode_lnvar = nn.Linear(50, self.latent_dim)
#         self.encode_drop  = nn.Dropout()

#         self.decode_net = conv.BeatDeconvMLP(n_channels=n_channels,
#                                              n_samples=n_samples,
#                                              n_latents=self.latent_dim)

#     def decode(self, z):
#         return self.decode_net(z)

#     def encode(self, x):
#         h1 = self.encode_net(x).view(x.size(0), -1)
#         h1 = self.erelu(h1)
#         return self.encode_mu(h1), self.encode_lnvar(h1)

#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = Variable(std.data.new(std.size()).normal_())
#         return eps.mul(std).add_(mu)


class LinearVAE(VAE):
    def __init__(self, **kwargs):
        super(LinearVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)
        self.verbose    = kwargs.get("verbose", False)

        # construct generative network for beats
        self.decode_net = nn.Linear(self.latent_dim, self.data_dim)

        # encode network
        self.encode_mu = nn.Linear(self.data_dim, self.latent_dim)
        self.encode_lnvar = nn.Linear(self.data_dim, self.latent_dim)

        # z parameters --- add mean
        self.init_params()

    def decode(self, z):
        return self.decode_net(z).view(-1, *self.data_shape)

    def encode(self, x):
        x = x.view(-1, self.data_dim)
        return self.encode_mu(x), self.encode_lnvar(x)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar


class LinearCycleVAE(LinearVAE):
    """ Reconstructs a high-dimensional observation using an additional
    reconstruction penalty """
    def set_discrim_model(self, discrim_model, discrim_beta):
        self.discrim_model = [discrim_model]
        self.discrim_beta = discrim_beta

    def lossfun(self, data, recon_data, target, mu, logvar):
        # vae ELBO loss
        vae_loss = super(LinearCycleVAE, self).lossfun(
            data, recon_data, target, mu, logvar)

        if self.discrim_beta == 0:
            return vae_loss

        # discrim reconstruction loss
        zdiscrim_data  = self.discrim_model[0](data)
        zdiscrim_recon = self.discrim_model[0](recon_data)
        disc_loss = self.discrim_beta * \
            torch.sum((zdiscrim_data-zdiscrim_recon)**2)
        #print(disc_loss, vae_loss)
        assert ~np.isnan(vae_loss.clone().detach().cpu())
        assert ~np.isnan(disc_loss.clone().detach().cpu())
        return vae_loss + disc_loss


class BeatMlpVAE(VAE):

    def __init__(self, **kwargs):
        super(BeatMlpVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.hdims      = kwargs.get("hdims", [500])
        self.verbose    = kwargs.get("verbose", False)
        self.sigmoid_out = kwargs.get("sigmoid_output", False)
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)

        # construct generative network for beats
        sizes = [self.latent_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout())

        modules.append(nn.Linear(sizes[-1], self.data_dim))
        if self.sigmoid_out:
            modules.append(nn.Sigmoid())
        else:
            modules.append(nn.Tanh())
        self.decode_net = nn.Sequential(*modules)

        # encoder network guts (reverses the generative process)
        rsizes = self.hdims[::-1]
        emodules = [nn.Linear(self.data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout())

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        self.init_params()

    def decode(self, z):
        return self.decode_net(z).view(-1, *self.data_shape)

    def encode(self, x):
        h1 = self.encode_net(x.view(-1, self.data_dim))
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

class BeatMlpCycleVAE(BeatMlpVAE):
    """ Reconstructs a high-dimensional observation using an additional
    reconstruction penalty """
    def set_discrim_model(self, discrim_model, discrim_beta):
        self.discrim_model = [discrim_model]
        self.discrim_beta = discrim_beta

    def lossfun(self, data, recon_data, target, mu, logvar):
        # vae ELBO loss
        vae_loss = super(BeatMlpCycleVAE, self).lossfun(
            data, recon_data, target, mu, logvar)

        if self.discrim_beta == 0:
            return vae_loss

        # discrim reconstruction loss
        zdiscrim_data  = self.discrim_model[0](data)
        zdiscrim_recon = self.discrim_model[0](recon_data)
        disc_loss = self.discrim_beta * \
            torch.sum((zdiscrim_data-zdiscrim_recon)**2)

        assert ~np.isnan(vae_loss.clone().detach().cpu())
        assert ~np.isnan(disc_loss.clone().detach().cpu())
        return vae_loss + disc_loss


class BeatMlpCondVAE(VAE):
    """ Reconstructs a high-dimensional observation given a conditioning
    latent variable (could be a sample) """
    def __init__(self, **kwargs):
        super(BeatMlpCondVAE, self).__init__(**kwargs)
        n_channels      = kwargs.get("n_channels")  # 3 for long leads
        n_samples       = kwargs.get("n_samples")   # 70-ish for single beats
        self.latent_dim = kwargs.get("latent_dim")
        self.cond_dim   = kwargs.get("cond_dim")
        unsupervised_dropout_p = kwargs.get("unsupervised_dropout_p", .25)
        self.hdims      = kwargs.get("hdims", [500])
        self.data_dim   = n_channels * n_samples
        self.data_shape = (n_channels, n_samples)

        # construct generative network for beats: dim(zfull) => data_dim
        self.zu_dropout = nn.Dropout(p=unsupervised_dropout_p)
        sizes   = [self.latent_dim + self.cond_dim] + self.hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=.25))

        self.decode_net = nn.Sequential(*modules)
        self.decode_final = nn.Linear(sizes[-1]+self.latent_dim+self.cond_dim, self.data_dim)

        # encoder network guts (reverses the generative process)
        rsizes = self.hdims[::-1]
        emodules = [nn.Linear(self.data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout(p=.25))

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], self.latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        self.init_params()

    def set_discrim_model(self, discrim_model):
        self.discrim_model = [discrim_model]

    def decode(self, z):
        h = self.decode_net(z)
        h = self.decode_final(torch.cat([z, h], 1))
        return h.view(-1, *self.data_shape)

    def encode(self, x):
        h1 = self.encode_net(x)
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        zcond = self.discrim_model[0](x)
        # do dropout on the unsupervised part to encourage use
        # of the conditional code
        zfull = torch.cat([zcond, self.zu_dropout(z)], 1)
        return self.decode(zfull), zfull, mu, logvar


###################
# Loss functions  #
###################
def mse_loss(recon_x, x):
    return torch.sum((recon_x - x) ** 2, dim=1)

def recon_loglike_function(recon_x, x, noise_var=.1*.1): # was .1
    num_obs_per_batch = x.shape[1]
    ln_noise_var = np.log(noise_var)
    diff = x - recon_x
    ll   = -(.5/noise_var) * (diff*diff).sum(1) \
            -(.5*ln2pi + .5*ln_noise_var) * num_obs_per_batch
    return ll

def binary_recon_loglike_function(recon_x, x):
    ll = F.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='none').sum(dim=-1)
    return -1.*ll

def kldiv_to_std_normal(mu, logvar):
    # KL(q(z) || p(z)) where q(z) is the recognition network normal
    KL_q_to_prior = .5*torch.sum(logvar.exp() - logvar + mu.pow(2) - 1, dim=1)
    return KL_q_to_prior


##################
# Models Modules #
##################

class MvnVAE(VAE):

    def __init__(self, latent_dim, data_dim, hdims=[500]):
        super(MvnVAE, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim   = data_dim

        # construct generative network for beats
        sizes = [latent_dim] + hdims
        modules = []
        for din, dout in zip(sizes[:-1], sizes[1:]):
            modules.append(nn.Linear(din, dout))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout())

        modules.append(nn.Linear(sizes[-1], data_dim))
        modules.append(nn.Tanh())
        self.decode_net = nn.Sequential(*modules)

        # encoder network guts (reverses the generative process)
        rsizes = hdims[::-1]
        emodules = [nn.Linear(data_dim, sizes[-1])]
        for din, dout in zip(rsizes[:-1], rsizes[1:]):
            emodules.append(nn.Linear(din, dout))
            emodules.append(nn.ReLU())
            emodules.append(nn.Dropout())

        self.encode_net   = nn.Sequential(*emodules)
        self.encode_mu    = nn.Linear(rsizes[-1], latent_dim)
        self.encode_lnvar = nn.Linear(rsizes[-1], latent_dim)
        self.encode_drop  = nn.Dropout()

        # z parameters --- add mean
        #self.zmean = nn.Parameter(torch.randn(latent_dim)*.01)
        #self.class_layer = nn.Linear(latent_dim, 1)
        #self.class_layer = nn.Linear(1, 1)
        self.init_params()

    def name(self):
        return "model-mvn-vae"

    def decode(self, z):
        return self.decode_net(z)
        #.view(-1, self.num_channels, self.num_samples)

    def encode(self, x):
        h1 = self.encode_net(x)
        return self.encode_mu(h1), self.encode_lnvar(h1)

    def sample_and_decode(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def reparameterize(self, mu, logvar, training=False):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, use_mean=False):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        if use_mean:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def lossfun(self, data, recon_data, target, mu, logvar):
        recon_ll = recon_loglike_function(recon_data, data) #) / data.size(1)
        kl_to_prior = kldiv_to_std_normal(mu, logvar) #a) / data.size(1)
        return -torch.mean(recon_ll - kl_to_prior)

    def kl_q_to_prior():
        pass

def decode_batch_list(mod, zmat, batch_size=256):
    data = torch.utils.data.TensorDataset(
        torch.FloatTensor(zmat), torch.FloatTensor(zmat))
    loader = torch.utils.data.DataLoader(data,
        batch_size=batch_size, shuffle=False, pin_memory=True)
    batch_res = []
    if torch.cuda.is_available():
        mod.cuda()
        do_cuda = True
    for batch_idx, (data, target) in enumerate(pyprind.prog_bar(loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()

        res = mod.decode(data)
        batch_res.append(res.data.cpu())

    return torch.cat(batch_res, dim=0)
