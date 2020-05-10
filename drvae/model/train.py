import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)

import os, sys #; sys.path.append("../experiments-pause")
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
sns.set_palette(sns.color_palette("Set2", 10))
torch.autograd.set_detect_anomaly(True)

import numpy as np
ln2pi = np.log(2*np.pi)

def fit_vae(model, Xtrain, Xval, Xtest, Ytrain, Yval, Ytest, **kwargs):
    # args
    #kwargs = {}
    do_cuda       = kwargs.get("do_cuda", torch.cuda.is_available())
    batch_size    = kwargs.get("batch_size", 1024)
    epochs        = kwargs.get("epochs", 40)
    output_dir    = kwargs.get("output_dir", "./")
    #discrim_model = kwargs.get("discrim_model", None)
    cond_dim      = kwargs.get("cond_dim", 1)
    zu_dropout_p  = kwargs.get("zu_dropout_p", .2)
    log_interval  = kwargs.get("log_interval", None)
    learning_rate = kwargs.get("lr", 1e-3)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 25)
    epoch_log_interval = kwargs.get("epoch_log_interval", 1)
    plot_interval      = kwargs.get("plot_interval", 10)
    dataset = kwargs.get("dataset", None)
    torch_seed = kwargs.get("torch_seed", None)
    scale_down_image_loss = kwargs.get("scale_down_image_loss", 
                                       False)
    anneal_rate = kwargs.get("anneal_rate", 0.005)
    num_zero_kl_epochs = kwargs.get("num_zero_kl_epochs", 20)
    #data_dim   = Xtrain.shape[1]
    print("-------------------")
    print("fitting vae: ", kwargs)
    
    # set up a dataset object
    if dataset is not None:
       # torchxrayvision map-syle dataset
       print('dataset is xray.')
       train_size = int(0.8 * len(dataset))
       valid_size = len(dataset) - train_size
       torch.manual_seed(torch_seed)
       train_data, val_data = torch.utils.data.random_split(
           dataset, [train_size, valid_size])
       train_epoch_func = train_epoch_xraydata
       test_epoch_func = test_epoch_xraydata
    else: 
        # andy's version - input data to func
        train_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(Xtrain), torch.FloatTensor(Ytrain[:,None]))
        val_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(Xval), torch.FloatTensor(Yval[:,None]))
        train_epoch_func = train_epoch
        test_epoch_func = test_epoch
    # below applies for both
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, num_workers=16, pin_memory=True, 
            drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
        drop_last=True,
        batch_size=batch_size, num_workers=16, pin_memory=True, shuffle=True)

    # create model
    if do_cuda:
        print('doing cuda')
        model.cuda()
        model.is_cuda = True

    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(" training %d param groups"%len(plist))
    optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=1e-5)

    # main training loop
    best_val_loss, best_val_state = np.inf, None
    prev_val_loss = np.inf
    train_elbo = []
    train_kl = []
    train_disc_loss = []
    val_elbo = []
    val_kl = []
    val_disc_loss = []
    print("{:10}  {:10}  {:10}  {:10}  {:10}  {:10}".format(
        "Epoch", "train-loss", "val-loss", "train-rmse", "val-rmse", "train/val-p"))
    for epoch in range(1, epochs + 1):
        
        # set beta for kl loss, start with 20 epochs with 0 loss, than gradually
        # add the KL term
        kl_beta = kl_schedule(epoch, num_zero_kl_epochs, anneal_rate) 

        # train/val over entire dataset
        tloss, trmse, tprecon, t_kl_loss, t_disc_loss = train_epoch_func(epoch, model, train_loader,
                                            optimizer     = optimizer,
                                            do_cuda       = do_cuda,
                                            log_interval  = log_interval,
                                            num_samples   = 1,
                                            scale_down_image_loss=scale_down_image_loss,
                                            kl_beta = kl_beta)
        
        vloss, vrmse, vprecon, v_kl_loss, v_disc_loss = test_epoch_func(epoch, model, 
                                    val_loader, do_cuda, 
                                    scale_down_image_loss=scale_down_image_loss,
                                    kl_beta = kl_beta)
        
        if epoch % epoch_log_interval == 0:

            print("{:10}  {:10}  {:10}  {:10}  {:10}  {:10}".format(
              epoch, "%2.4f"%tloss, "%2.4f"%vloss, 
              "%2.4f"%trmse, "%2.4f"%vrmse,
              "%2.3f / %2.3f"%(tprecon, vprecon)))

            # track elbo values
            train_elbo.append(tloss)
            train_kl.append(t_kl_loss)
            train_disc_loss.append(t_disc_loss)
            val_elbo.append(vloss)
            val_kl.append(v_kl_loss)
            val_disc_loss.append(v_disc_loss)

        # keep track of best model by validation rmse
        if vrmse < best_val_loss:
            best_val_loss, best_val_state = vrmse, model.state_dict()

        # update learning rate if we're not doing better
        if epoch % lr_reduce_interval == 0: # 100 and vloss >= prev_val_loss:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= .5

        # visualize reconstruction
        if epoch % plot_interval == 0: # add back the saved reconstruction
            # save_reconstruction(model, train_data,
            #     output_file = os.path.join(output_dir, "train-recon-%d.png"%epoch))
            # save_reconstruction(model, val_data,
            #     output_file = os.path.join(output_dir, "val-recon-%d.png"%epoch))
            save_elbo_plots(train_elbo, val_elbo, output_dir)
            #torch.save({'train_elbo' : train_elbo,
            #            'val_elbo'   : val_elbo,
            #            'state_dict' : model.state_dict(),
            #            'optimizer'  : optimizer.state_dict()},
            #            f=os.path.join(output_dir, model.name() + ".pth.tar"))

    # load in best state by validation loss
    model.load_state_dict(best_val_state)
    model.eval()

    resdict = {'train_elbo'   : train_elbo,
               'train_kl' : train_kl,
               'train_disc_loss': train_disc_loss,
               'val_elbo'     : val_elbo,
               'val_kl' : val_kl,
               'val_disc_loss': val_disc_loss,
               #'model_state'  : model.state_dict(),
               #'data_dim'     : data_dim,
               'cond_dim'     : cond_dim,
               'discrim_model': None,
               #'latent_dim'   : latent_dim,
               'zu_dropout_p' : zu_dropout_p,
               'output_dir'   : output_dir,
               #'hdims'        : hdims
               }
    return resdict



################################################################
# Training functions --- should work with both BeatVAE types   #
################################################################
def train_epoch(epoch, model, train_loader,
                optimizer     = None,
                do_cuda       = True,
                log_interval  = None,
                num_samples   = 1):

    # set up train/eval mode
    do_train = False if optimizer is None else True
    if do_train:
        model.train()
    else:
        model.eval()

    # run thorugh batches in data loader
    train_loss = 0
    recon_rmse = 0.
    recon_prob_sse, recon_z_sse = 0., 0.
    trues, preds = [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()

        if do_train:
            optimizer.zero_grad()

        # compute encoding distribution
        loss = 0
        for _ in range(num_samples):
            recon_batch, z, mu, logvar = model(data)
            recon_batch = recon_batch.view_as(data)
            # model computes its own loss
            loss += model.lossfun(data, recon_batch, target, mu, logvar)

        if do_train:
            loss.backward()
            optimizer.step()

        # track pred probs
        recon_rmse += torch.std(recon_batch-data).data.item()*data.shape[0]
        train_loss += loss.data.item()*data.shape[0]

        # if we have a discrim model, track how well it's reconstructing probs
        if hasattr(model, "discrim_model"):
            zrec = model.discrim_model[0](recon_batch)
            zdat = model.discrim_model[0](data)
            prec = torch.sigmoid(zrec)
            pdat = torch.sigmoid(zdat)
            recon_z_sse += torch.var(zrec-zdat).data.item()*data.shape[0]
            recon_prob_sse += torch.var(prec-pdat).data.item()*data.shape[0]

        if (log_interval is not None) and (batch_idx % log_interval == 0):
            print('  epoch: {epoch} [{n}/{N} ({per:.0f}%)]\tLoss: {loss:.6f}'.format(
                epoch = epoch,
                n     = batch_idx * train_loader.batch_size,
                N     = len(train_loader.dataset),
                per   = 100. * batch_idx / len(train_loader),
                loss  = loss.data.item() / len(data)))

    # compute average loss
    N = len(train_loader.dataset)
    return train_loss/N, recon_rmse/N, np.sqrt(recon_prob_sse/N)


def test_epoch(epoch, model, data_loader, do_cuda):
    return train_epoch(epoch, model, train_loader=data_loader,
                       optimizer=None, do_cuda=do_cuda)

def train_epoch_xraydata(epoch, model, train_loader,
                optimizer     = None,
                do_cuda       = True,
                log_interval  = None,
                num_samples   = 1,
                scale_down_image_loss = False,
                kl_beta = 1.0):
    '''This function is identical to train_epoch except that it uses 
    a map style dataset from torchxrayvision'''
    # set up train/eval mode
    do_train = False if optimizer is None else True
    if do_train: 
        model.train()
        if hasattr(model, "discrim_model"): # set discrim_model to eval mode, even when training.
            model.discrim_model.eval()
    else:
        model.eval()
    
    # run thorugh batches in data loader
    train_loss = 0
    recon_rmse = 0.
    kl_loss = 0.
    discrim_loss = 0.
    recon_prob_sse, recon_z_sse = 0., 0.
    loss_list = [] # renewed every epoch for debugging
    latent_loss_list = []
    #trues, preds = [], []
    #t = tqdm(train_loader) # and also had t instead of train loader inside brackets.
    for batch_idx, samples in enumerate(train_loader):
        data = samples["img"].float()
        target = samples["lab"]

        if do_cuda:
            data, target = data.cuda(), target.cuda()

        if do_train:
            optimizer.zero_grad()

        # compute encoding distribution
        loss = 0
        # ToDo: num_samples currently not supported, loss isn't accumulated across samples
        for _ in range(num_samples): 
            recon_batch, z, mu, logvar = model(data)

            # model computes its own loss. 
            # loss is a tuple and we minimize its first element.
            # add KL annealing to the DRVAE function as well that is a func of epoch
            loss = model.lossfun(
                                  data, recon_batch, 
                                  target, mu, logvar,
                                  kl_beta,
                                  scale_down_image_loss)   
            ## previous version.
            # loss += model.lossfun(data, recon_batch, 
            #           target, mu, logvar,
            #           kl_beta,
            #           scale_down_image_loss) 
            
            loss_list.append(loss[0].detach().cpu().numpy()) # loss list within a an epoch.
            latent_loss_list.append(loss[2].detach().cpu().numpy())
            # tests 
            # for i in range(data.shape[0]):
            #     if(len(data[i,0,:,:].flatten().unique())<2):
            #         print('image %s' %str(i))
            #         print(data[i,0,:,:].flatten().unique())
            #     if(len(recon_batch[i,0,:,:].flatten().unique())<2):
            #         print('recon image %s' %str(i))
            #         print(recon_batch[i,0,:,:].flatten().unique())

            
            if batch_idx>1:
                if np.abs(loss_list[-1]/loss_list[-2]) > 100.00 or \
                    np.abs(latent_loss_list[-1]/latent_loss_list[-2]) > 100.00:
                    print('------loss just jumped!---------')
                    plot_recon_batch(recon_batch, data, epoch, batch_idx)
                    plot_bottleneck_stats(mu, logvar, z, epoch, batch_idx, bins = 100)
                    torch.save(recon_batch, 'prob_recon_epoch%i_batch_%i.pt' %(epoch, batch_idx))
                    torch.save(data, 'prob_data_epoch%i_batch_%i.pt' %(epoch, batch_idx))
                    torch.save(mu, 'prob_mu_epoch%i_batch_%i.pt' %(epoch, batch_idx))
                    torch.save(logvar, 'prob_logvar_epoch%i_batch_%i.pt' %(epoch, batch_idx))
                    print('tensors, reconstructions and latent stats saved in %s' % \
                          os.getcwd())

                    # for i in range(data.shape[0]):
                    #     print('data image %s' %str(i))
                    #     print(data[i,0,:,:].flatten().unique())
                    #     print(len(data[i,0,:,:].flatten().unique()))
                    #     print('recon image %s' %str(i))
                    #     print(recon_batch[i,0,:,:].flatten().unique())
                    #     print(len(recon_batch[i,0,:,:].flatten().unique()))
                    #     print('mu %s' % str(i))
                    #     print(mu[i,:].flatten())
                    # plot and save
                   
                        #print(data)

            if np.sum(np.isnan(data.detach().cpu().numpy().flatten())) !=0 or \
                (data.view(data.shape[0],-1).sum(dim=1).detach().cpu().numpy() == 0).any() or \
                np.sum(np.isnan(recon_batch.detach().cpu().numpy().flatten())) !=0 or \
                np.sum(np.isnan(mu.detach().cpu().numpy().flatten())) !=0 or \
                np.sum(np.isnan(logvar.detach().cpu().numpy().flatten())) !=0:
                
                print('data is nan:')
                print(np.sum(np.isnan(data.detach().cpu().numpy().flatten())))
                print(np.min(data.detach().cpu().numpy().flatten()))
                print(np.max(data.detach().cpu().numpy().flatten()))
                print('data raw:')
                print(data)
                print('recon is nan:')
                print(np.sum(np.isnan(recon_batch.detach().cpu().numpy().flatten())))
                print(np.min(recon_batch.detach().cpu().numpy().flatten()))
                print(np.max(recon_batch.detach().cpu().numpy().flatten()))
                print('mu is nan:')
                print(np.sum(np.isnan(mu.detach().cpu().numpy().flatten())))
                print(mu.detach().cpu().numpy().flatten())
                print(np.min(mu.detach().cpu().numpy().flatten()))
                print(np.max(mu.detach().cpu().numpy().flatten()))
                print('lnvar is nan:')
                print(np.sum(np.isnan(logvar.detach().cpu().numpy().flatten())))
                print(logvar.detach().cpu().numpy().flatten())
                print(np.min(logvar.detach().cpu().numpy().flatten()))
                print(np.max(logvar.detach().cpu().numpy().flatten()))


        if do_train:
            # save last batch of data in case it crashes. can remove when runs smoothly.
            torch.save(recon_batch, 'recon_batch.pt' %(epoch, batch_idx))
            torch.save(data, 'recon_batch.pt' %(epoch, batch_idx))
            torch.save(mu, 'mu_batch.pt' %(epoch, batch_idx))
            torch.save(logvar, 'logvar_batch.pt' %(epoch, batch_idx))
            torch.save(logvar, 'z_batch.pt' %(epoch, batch_idx))

            # minimize first element in the loss tuple.
            loss[0].backward() 
            optimizer.step()
        
        # add errors from each batch to the total train_loss of epoch
        recon_rmse += torch.std(recon_batch-data).data.item()*data.shape[0]
        train_loss += loss[0].data.item()*data.shape[0]
        kl_loss += loss[2].data.item()*data.shape[0]
        discrim_loss += (loss[0].data.item()*data.shape[0] - 
                         loss[1].data.item()*data.shape[0])

        # if we have a discrim model, track how well it's reconstructing probs
        # ToDo: current assumption is that the output is on the logit scale, also in the loss
        # function. if decide to do otherwise, change here and in loss.
        if hasattr(model, "discrim_model"):
            zrec = model.discrim_model(recon_batch)[:, model.dim_out_to_use]
            zdat = model.discrim_model(data)[:, model.dim_out_to_use]
            # note, sigmoid on the output of the model.
            prec = torch.sigmoid(zrec)
            pdat = torch.sigmoid(zdat)
            recon_z_sse += torch.var(zrec-zdat).data.item()*data.shape[0]
            recon_prob_sse += torch.var(prec-pdat).data.item()*data.shape[0]

        if (log_interval is not None) and (batch_idx % log_interval == 0):
            print(
                '  epoch: {epoch} [{n}/{N} ({per:.0f}%)]\tLoss: {total_loss:.4f}, Disc. Loss: {discrim_loss:.4f}, Lat. Loss: {latent_loss:.4f}, Recon. Loss: {image_loss:.4f}'.format(
                epoch = epoch,
                n     = batch_idx * train_loader.batch_size,
                N     = len(train_loader.dataset),
                per   = 100. * batch_idx / len(train_loader),
                total_loss  = loss[0].data.item() / len(data),
                discrim_loss = loss[0].data.item() / len(data) - 
                    loss[1].data.item() / len(data), # DRVAE - VAE
                latent_loss = loss[2].data.item() / len(data),
                image_loss = loss[3].data.item() / len(data),
                ))
                

    # compute average loss
    N = len(train_loader.dataset)
    return train_loss/N, recon_rmse/N, np.sqrt(recon_prob_sse/N), kl_loss/N, discrim_loss/N

def test_epoch_xraydata(epoch, model, data_loader, 
                        do_cuda, scale_down_image_loss, kl_beta):
    return train_epoch_xraydata(epoch, model, train_loader=data_loader,
                       optimizer=None, do_cuda=do_cuda, 
                       scale_down_image_loss=scale_down_image_loss,
                       kl_beta = kl_beta)


#def batch_reconstruct(mod, X):
#    data = torch.utils.data.TensorDataset(
#        torch.FloatTensor(X), torch.FloatTensor(X))
#    loader = torch.utils.data.DataLoader(data,
#        batch_size=batch_size, shuffle=False, pin_memory=True)
#    batch_res = []
#    if torch.cuda.is_available():
#        mod.cuda()
#        do_cuda = True
#    for batch_idx, (data, target) in enumerate(pyprind.prog_bar(loader)):
#        data, target = Variable(data), Variable(target)
#        if do_cuda:
#            data, target = data.cuda(), target.cuda()
#            data, target = data.contiguous(), target.contiguous()
#
#        recon_batch, z, mu, logvar = model(data)
# 
#        res = mod.forward(data)
#        batch_res.append(res.data.cpu())
#
#    return torch.cat(batch_res, dim=0)
#

def kl_schedule(epoch_num, n_epochs_with_zero, rate=0.001):
    """
    Anneal the KL term in the VAE cost function

    With rate 0.001 and n_epochs_with_zero = 20, we need 1020 epochs to get
    kl_lambda = 1

    Args:
        epoch_num:
        n_epochs_with_zero:
        rate:

    Returns:
        float

    """
    if epoch_num < n_epochs_with_zero:
        kl_lambda = 0.0
    else:
        kl_lambda = np.minimum((epoch_num - n_epochs_with_zero) * rate, 1)
    return kl_lambda


#######################
# plotting functions  #
#######################

def save_elbo_plots(train_elbo, val_elbo, output_dir):
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    ax.plot(-np.array(train_elbo), label="train elbo")
    ax.plot(-np.array(val_elbo), label="val elbo")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "elbo-by-epoch.png"), bbox_inches='tight')
    plt.close("all")

# my version # to be adapted. 
def save_reconstruction(model, dataset, output_file):
    model_is_cuda = next(model.parameters()).is_cuda

    # select random data
    data_tensor, _ = dataset.tensors
    n_data = data_tensor.shape[0]
    idx    = np.random.permutation(n_data)[:10]

    # reconstruct data
    data_npy = data_tensor.numpy()[idx]
    data = Variable(torch.FloatTensor(data_npy)).contiguous()
    if model_is_cuda:
        data = data.cuda()
    recon_x, z, mu, logvar = model(data)

    # reshape data and recon
    #if len(data.shape) == 2:
    #    data = data.view(-1, 3, n_dim//3)
    #    recon_x = recon_x.view_as(data)

    # plot beat and save
    X = recon_x.detach()
    if model_is_cuda:
        X = X.cpu()
    X = X.numpy()
    fig, ax = plt.figure(figsize=(8,6)), plt.gca()
    plot_images(X[:10,:], ax=ax)

    fig.savefig(output_file, bbox_inches='tight')
    plt.close("all")
    
# # Andy's version below
# def save_reconstruction(model, dataset, output_file):
#     model_is_cuda = next(model.parameters()).is_cuda

#     # select random data
#     data_tensor, _ = dataset.tensors
#     n_data = data_tensor.shape[0]
#     idx    = np.random.permutation(n_data)[:10]

#     # reconstruct data
#     data_npy = data_tensor.numpy()[idx]
#     data = Variable(torch.FloatTensor(data_npy)).contiguous()
#     if model_is_cuda:
#         data = data.cuda()
#     recon_x, z, mu, logvar = model(data)

#     # reshape data and recon
#     #if len(data.shape) == 2:
#     #    data = data.view(-1, 3, n_dim//3)
#     #    recon_x = recon_x.view_as(data)

#     # plot beat and save
#     X = recon_x.detach()
#     if model_is_cuda:
#         X = X.cpu()
#     X = X.numpy()
#     fig, ax = plt.figure(figsize=(8,6)), plt.gca()
#     plot_images(X[:10,:], ax=ax)

#     fig.savefig(output_file, bbox_inches='tight')
#     plt.close("all")

def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=None, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    import matplotlib.cm
    import matplotlib.pyplot as plt; plt.ion()
    if cmap is None:
        cmap = matplotlib.cm.binary
    N_images = images.shape[0]
    N_rows = (N_images - 1) // ims_per_row + 1
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    #ax.figure.patch.set_visible(False)
    ax.patch.set_visible(False)
    return cax

def plot_beat(data, recon_x=None):
    # first, move everything to CPU
    if data.is_cuda:
        data = data.cpu()
    if recon_x is not None and recon_x.is_cuda:
        recon_x = recon_x.cpu()

    # plot a bunch of data
    ncol = int(data.shape[0] / 2)
    nrow = 2
    fig, axarr = plt.subplots(nrow, ncol, figsize=(4*ncol, 2*nrow))
    for i, ax in enumerate(axarr.flatten()):
        ax.plot(data[i, 0, :].data.numpy(), "-o", label="beat")
        if recon_x is not None:
            ax.plot(recon_x[i, 0, :].data.numpy(), label="recon")
        ax.legend()
    return fig, axarr

def plot_bottleneck_stats(mu, logvar, z, epoch, batch_idx, bins = 100):
    f, axarr = plt.subplots(1,3, figsize = (12,4))
    mu_np = mu.detach().cpu().numpy().flatten()
    logvar_np = logvar.detach().cpu().numpy().flatten()
    z_np = z.detach().cpu().numpy().flatten()

    axarr[0].hist(mu_np, bins=100, density = True);
    axarr[0].set_title('mu [%.2f, %.2f]' % (np.min(mu_np), np.max(mu_np)));
    axarr[1].hist(logvar.detach().cpu().numpy().flatten(), bins=100, density = True);
    axarr[1].set_title('logvar [%.2f, %.2f]' % (np.min(logvar_np), np.max(logvar_np)));
    axarr[2].hist(z.detach().cpu().numpy().flatten(), bins=100, density = True);
    axarr[2].set_title('z [%.2f, %.2f]' % (np.min(z_np), np.max(z_np)));
    f.suptitle('Epoch %i, Batch %i stats' %(epoch, batch_idx))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('epoch_%i_batch_%i_latent_stats.png' % (epoch, batch_idx))
    
def plot_recon_batch(recon, data, epoch, batch_idx):
    """"wrapper around plot_images."""
    recon = recon.detach().cpu().numpy()
    data = data.detach().cpu().numpy()
    batch_size = data.shape[0]
    image_size = data.shape[-1] # squared images.
    f, axarr = plt.subplots(1,2, figsize = (20,17))
    axarr[0].set_title('recon.',  fontsize = 24)
    axarr[1].set_title('data',  fontsize = 24)
    plot_images(recon.squeeze(1).reshape(batch_size,-1), axarr[0], ims_per_row=5, 
                padding=5, digit_dimensions=(image_size, image_size),
                cmap='gray', vmin=None, vmax=None)
    plot_images(data.squeeze(1).reshape(batch_size,-1), axarr[1], ims_per_row=5, 
                padding=5, digit_dimensions=(image_size, image_size),
                cmap='gray', vmin=None, vmax=None)
    f.suptitle('recons: epoch %i batch %i' % (epoch, batch_idx), fontsize = 28)
    f.tight_layout()
    plt.savefig('epoch_%i_batch_%i_recon_stats.png' % (epoch, batch_idx))

                
