# Wasserstein GAN with gradient penalty
# Code for training of WGAN-gp

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch import autograd

class Trainer():
    def __init__(self, model_g, optim_g, model_c, optim_c, model_e, epochs, device, dataloader,in_header):

        self.generator = model_g    # Generator
        self.opt_g = optim_g        # Optimizer for Generator
        self.critic = model_c       # Critic
        self.opt_c = optim_c        # Optimizer for Critic
        self.estimator = model_e    # Estimator for specific surface area
        self.n_epoch = epochs       # Number of epochs
        self.device = device
        self.dataloader = dataloader    # Dataloader
        self.header = in_header         # Header of input data
        self.losses = {"G_loss":[], "D_loss":[], "gp_loss":[], "Wasser_d":[], "vf_loss":[], "ssa_loss":[]} # Losses

        # ----- Hyper Parameters ----- #
        self.w_gp = 10          # Weight for gradient penalty
        self.w_param = 1000     # Weight for volume fraction and specific surface area
        self.n_critic = 1       # Number of critic iteration
        self.save_epoch = 1     # Save model per epoch
        self.timing = 10        # Start to train with specific surface area
        self.latent_size = 100  # Latent size

    def _get_min_max(self):
        """ 
        Function of getting minimum and maximum value of specific surface area
        """
        path = os.path.join(self.header,"results.dat")
        df = pd.read_table(path, sep="\s+") 
        mm_list = [[df["SV{}".format(i)].min(), df["SV{}".format(i)].max()] for i in range(3)]
        return mm_list
    
    def _sample_generator(self, vf, ssa, b_len):
        """ 
        Function of generating structure data
        :param vf : volume fraction
        :param ssa : specific surface area
        :param b_len : length of minibatch
        """
        noise = torch.randn(b_len, self.latent_size, 4,4,4, device=self.device)
        g_data = self.generator(noise, vf, ssa)
        return g_data

    def _plot_losses(self,epoch,itr):
        """ 
        Function of plotting losses
        :param epoch : number of epoch
        :param itr : number of iteration
        """
        print("[{:03}/{:03}][{:03}/{:03}] G_loss: {:.04f} D_loss: {:.04f} Wasser_D: {:.04f} ".format(
            epoch+1, self.n_epoch, itr+1, len(self.dataloader), self.losses["G_loss"][-1], self.losses["D_loss"][-1], self.losses["Wasser_d"][-1]
        ))

        file = open("./log.dat","a")
        file.write("[{:03}/{:03}][{:03}/{:03}] G_loss: {:.04f} D_loss: {:.04f} Wasser_D: {:.04f} vf_loss: {:.04} ssa_loss: {:.04} gp_loss: {:.04} ".format(
            epoch+1, self.n_epoch, itr+1, len(self.dataloader), self.losses["G_loss"][-1], self.losses["D_loss"][-1], self.losses["Wasser_d"][-1],
            self.losses["vf_loss"][-1],self.losses["ssa_loss"][-1],self.losses["gp_loss"][-1]
        )+"\n")
        file.close()
        return 

    def _gradient_penalty(self, r_data, g_data):
        """ 
        Function of calculating gradient penalty
        :param r_data : real structure data
        :param g_data : generated structure data
        """

        is_cuda = torch.cuda.is_available()
        b_len = r_data.shape[0]
        epsilon = torch.rand(b_len,1,1,1,1)
        epsilon = epsilon.to(self.device)

        interpolated_img = epsilon * r_data + (1-epsilon) * g_data
        interpolated_out = self.critic(interpolated_img)

        grads = autograd.grad(outputs=interpolated_out,
                              inputs=interpolated_img,
                              grad_outputs=torch.ones(interpolated_out.shape).cuda() if is_cuda else torch.ones(interpolated_out.shape),
                              create_graph=True,
                              retain_graph=True)[0]
        
        grads = grads.reshape([b_len,-1])
        grad_penalty = ((grads.norm(2, dim=1)-1)**2).mean()
        self.losses["gp_loss"].append(grad_penalty.item())
        return grad_penalty

    def _loss_vf(self, b_len, g_data, in_vf, n_size=64):
        """ 
        Function of calculating loss of volume fraction
        :param b_len : length of minibatch
        :param g_data : generated structure data
        :param in_vf : input volume fraction
        :param n_size : voxel size of structure data
        """

        total_loss = 0.
        mass = n_size ** 3

        for i in range(b_len):
            struc = g_data[i]
            vf = in_vf[i]

            # ----- Calculate Mean Square Error ----- #
            Ni_loss = torch.square(torch.sum(struc[0,:,:,:])/mass - vf[0,0,0,0])
            YSZ_loss = torch.square(torch.sum(struc[1,:,:,:])/mass - vf[1,0,0,0])
            Pore_loss = torch.square(torch.sum(struc[2,:,:,:])/mass - vf[2,0,0,0])
            losses = Ni_loss + YSZ_loss + Pore_loss
            total_loss += losses

        vf_loss = total_loss / b_len
        self.losses["vf_loss"].append(vf_loss)
            
        return vf_loss
    
    def _phase_pickup(self, images, phase):
        """ 
        Function of picking up specific phase from structure data
        :param images : structure data
        :param phase : specific phase
        """
        if phase == 'Ni':
            imgs = images[:,0]
        elif phase == 'YSZ':
            imgs = images[:,1]
        elif phase == 'Pore':
            imgs = images[:,2]
        imgs = imgs.reshape(imgs.shape[0],1,imgs.shape[1],imgs.shape[2],imgs.shape[3])
        return imgs

    def _preprocess(self, structure):
        """ 
        Function of preprocessing structure data
        :param structure : structure data
        """
        name = ["Ni","YSZ","Pore"]
        struc_list = [self._phase_pickup(structure,i) for i in name]
        struc_input = torch.cat(struc_list, dim=0)
        return struc_input

    def _standardize(self, raw_ssa, mm_list):
        """ 
        Function of standardizing specific surface area
        :param raw_ssa : raw specific surface area
        :param mm_list : minimum and maximum value of specific surface area
        """
        length = len(raw_ssa) // 3
        stand_list = []
        for i in range(3):
            raw = raw_ssa[(i*length):((i+1)*length)]
            stand_value = (raw - mm_list[i][0])/(mm_list[i][1]-mm_list[i][0])
            stand_list.append(stand_value)
        stand = torch.cat(stand_list, dim=0)

        return stand

    def _loss_ssa(self, b_len, g_data, in_ssa, mm_list):
        """ 
        Function of calculating loss of specific surface area
        :param b_len : length of minibatch
        :param g_data : generated structure data
        :param in_ssa : input specific surface area
        :param mm_list : minimum and maximum value of specific surface area
        """
        self.estimator = self.estimator.eval()
        with torch.no_grad():
            g_data = g_data.to(self.device)
            pred_raw = self.estimator(self._preprocess(g_data)).detach().clone()
            pred = self._standardize(pred_raw,mm_list)
        true_raw = self._preprocess(in_ssa).to(self.device)
        true = true_raw[:,0,0,0,0].detach().clone()

        loss = torch.sum(torch.square(true - pred)).requires_grad_()
        ssa_loss = loss / (b_len)
        self.losses["ssa_loss"].append(ssa_loss)

        return ssa_loss 

    def _fixed_labels(self):
        """ 
        Function of generating fixed noise, volume fraction, and specific surface area
        """
        fixed_noise = torch.randn(1,self.latent_size,4,4,4,device=self.device)

        fixed_vf = torch.zeros([1,3,4,4,4]).to(self.device)
        fixed_vf[0,0,:,:,:] = 0.5058; fixed_vf[0,1,:,:,:] = 0.1927; fixed_vf[0,2,:,:,:] = 0.3015

        fixed_ssa = torch.zeros([1,3,4,4,4]).to(self.device)
        fixed_ssa[0,0,:,:,:] = 0.1512; fixed_ssa[0,1,:,:,:] = 0.4985; fixed_ssa[0,2,:,:,:] = 0.1612

        return fixed_noise, fixed_vf, fixed_ssa

    def _oh_to_bmp(self,struc_oh):
        """ 
        Function of converting one-hot to bmp
        :param struc_oh : one-hot structure data
        """
        struc_oh = struc_oh.to('cpu').numpy().copy()
        struc_arr = np.array(struc_oh)
        
        # ------- Extract each layer ------- #
        layer_ni = struc_arr[0,:,:,:]
        layer_ysz = struc_arr[1,:,:,:]
        layer_pore = struc_arr[2,:,:,:]

        struc_max = np.max(struc_oh,axis=0).squeeze()

        # ------- Compare each layer and max value ------- #
        bool_ni = np.array([layer_ni == struc_max]).squeeze()
        num_ni = bool_ni.astype(np.float32) * 255

        bool_ysz = np.array([layer_ysz == struc_max]).squeeze()
        num_ysz = bool_ysz.astype(np.float32) * 127

        bool_pore = np.array([layer_pore == struc_max]).squeeze()
        num_pore = bool_pore.astype(np.float32) * 0

        bmp_img = num_ni + num_ysz + num_pore

        return bmp_img

    def _plot_image(self, epoch, noise, vf, ssa, n_size=64):
        """ 
        Function of plotting structure data
        :param epoch : number of epoch
        :param noise : noise data
        :param vf : volume fraction
        :param ssa : specific surface area
        :param n_size : voxel size of structure data
        """
        g = self.generator.eval()
        # ------- Generate structure ------- #
        with torch.no_grad():
            fake_structure = g(noise,vf,ssa).reshape(3,n_size,n_size,n_size).detach()
        
        arr_structure = self._oh_to_bmp(fake_structure)

        # ------- Save Image ------- #
        fig = plt.figure(figsize=(25,25))
        for i in range(n_size):
            ax = fig.add_subplot(8,8,i+1)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.imshow(arr_structure[i],cmap='gray')
            fig.tight_layout()
        
        dir_path = "./Process_Images"
        os.makedirs(dir_path,exist_ok=True)
        fig.savefig(os.path.join(dir_path,"epoch_{}.png".format(epoch+1)))
        plt.close("all")
        
        return

    def train(self):
        iters = 0
        G_loss = D_loss = torch.Tensor([0])
        f_noise, f_vf, f_ssa = self._fixed_labels()
        mm_list = self._get_min_max()
        
        for epoch in range(self.n_epoch):
            self.generator.train()
            self.critic.train()

            for itr, data in enumerate(self.dataloader):
                iters += 1
                
                r_data = data[0].to(self.device)                     # Real structure 
                r_vf   = data[1][:,:3,:,:,:].to(self.device)         # Real volume fraction
                r_ssa  = data[1][:,3:,:,:,:].to(self.device)         # Real specific surface area
                b_len  = r_data.size(0)                              # Length of minibatch
                g_data = self._sample_generator(r_vf,r_ssa,b_len)    # Generated structure

                # ------ Train of Critic ----- #
                self.opt_c.zero_grad()
                op_fake = self.critic(g_data.detach())               # output of critic for generated data
                op_real = self.critic(r_data.detach())               # output of critic for real data

                wasser_d = (op_real.mean() - op_fake.mean())         # Calculated wasserstein distance
                loss_gp = self._gradient_penalty(r_data,g_data)
                self.losses["Wasser_d"].append(wasser_d.item())
                self.losses["gp_loss"].append(loss_gp.item())

                D_loss = - wasser_d + self.w_gp * loss_gp
                self.losses["D_loss"].append(D_loss.item())
                D_loss.backward()
                self.opt_c.step()

                # ----- Train for generator ----- #
                if iters % self.n_critic == 0:
                    self.opt_g.zero_grad()
                    g_data = self._sample_generator(r_vf, r_ssa, b_len)
                    loss_vf = self._loss_vf(b_len,g_data,r_vf)
                    loss_ssa = self._loss_ssa(b_len,g_data, r_ssa, mm_list)

                    g_loss = - self.critic(g_data).mean()

                    if epoch>=self.timing:
                        G_loss = g_loss + self.w_param * (loss_vf + loss_ssa)
                    else:
                        G_loss = g_loss + self.w_param * (loss_vf)

                    self.losses["G_loss"].append(G_loss.item())
                    G_loss.backward()
                    self.opt_g.step()
            
                self._plot_losses(epoch, itr)
            
            if(epoch + 1) % self.save_epoch == 0:
                os.makedirs("./save_model", exist_ok=True)
                torch.save(self.generator.state_dict(),"./save_model/Generator_{:03}epoch.pth".format(epoch+1))
            
            self._plot_image(epoch, f_noise, f_vf, f_ssa)

        return

