import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss, SupConLoss
from pytorch_lightning.utilities import rank_zero_only

import torch.nn as nn
import torch.distributed as dist

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def kde_gaussian_kernel(x, x_data, bandwidth):
    """
    Compute the Gaussian kernel for Kernel Density Estimation
    
    Args:
        x (Tensor): Evaluation points
        x_data (Tensor): Data points
        bandwidth (float): Bandwidth (smoothing parameter)
    
    Returns:
        Tensor: Gaussian kernel evaluated at x for each data point in x_data
    """
    num_data = x_data.size(0)
    x = x.view(-1, 1)
    x_data = x_data.view(1, -1)
    
    kernel = torch.exp(-0.5 * ((x - x_data) / bandwidth)**2)
    kernel = kernel / (bandwidth * torch.sqrt(2 * torch.tensor(3.14159265358979)))
    
    return kernel.sum(dim=1) / num_data

def kde(x, x_data, bandwidth):
    """
    Compute the Kernel Density Estimation
    
    Args:
        x (Tensor): Evaluation points
        x_data (Tensor): Data points
        bandwidth (float): Bandwidth (smoothing parameter)
    
    Returns:
        Tensor: Kernel density estimate evaluated at x
    """
    kernel = kde_gaussian_kernel(x, x_data, bandwidth)
    return kernel

class VMFDistribution(nn.Module):
    def __init__(self, p, kappa=1.0):
        # p: channels
        super(VMFDistribution, self).__init__()
        self.p = p
        self.mu = nn.Parameter(torch.randn(p))
        self.z_mean = nn.Parameter(torch.zeros(p))
        # mu = torch.randn(channels)
        # self.mu = nn.Parameter(mu / mu.norm(dim=-1, keepdim=True))  # 保持mu单位长度
        self.kappa = nn.Parameter(torch.tensor(kappa))  # kappa浓度参数

    def modified_bessel_first_kind_(self, N=10):
        """
        Computes the modified Bessel function of the first kind I_(p/2-1)(z)
        using the series expansion up to N terms.
        
        Args:
        N (int, optional): Number of terms in the series expansion.

        Returns:
        torch.Tensor: The evaluated Bessel function at self.kappa.
        """
        nu = self.p / 2 - 1
        sum_terms = torch.zeros_like(self.kappa, device = self.kappa.device)

        # We calculate each term in the series
        for k in range(N):
            # (z/2)^(2k+nu)
            exponent = 2 * k + nu
            term = (self.kappa / 2) ** exponent
            
            # k! * Gamma(nu + k + 1)
            log_fact_k = torch.lgamma(torch.tensor(k + 1.0, device=self.kappa.device))  # log(k!)
            log_gamma_nu_k_1 = torch.lgamma(torch.tensor(nu + k + 1, device=self.kappa.device))       # log(Gamma(nu + k + 1))
            
            # Calculate term divided by the product of factorials
            term /= torch.exp(log_fact_k + log_gamma_nu_k_1)  # Exp to turn logs back to standard form
            if term == 0:
                break
            sum_terms += term

        return sum_terms

    def modified_bessel_first_kind(self):
        M = self.p
        current_bessels = torch.zeros(M+1, device=self.mu.device, dtype=torch.float64)  # 当前和下一个贝塞尔函数的值
        next_bessels = torch.zeros(M+1, device=self.mu.device, dtype=torch.float64)

        # 初始化
        current_bessels[M] = 1.0  # I_M(kappa)
        next_bessels[M] = 0.0  # I_{M+1}(kappa)
        
        # 逆递推
        for nu in range(M, 0, -1):
            next_bessels[nu-1] = (2 * nu / self.kappa) * current_bessels[nu] + next_bessels[nu]
        
        # 标准的 I_0(kappa) 需要独立计算，这里可以使用直接计算或查表
        I_0 = torch.i0(self.kappa)  # PyTorch 自带的零阶第一类修正贝塞尔函数
        normalized_bessels = next_bessels * (I_0 / next_bessels[0])
        
        # 返回需要的阶数
        nu = self.p // 2 - 1
        return normalized_bessels[nu]


    def compute_prob(self, x):
        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)
        # bessel_func_result = self.modified_bessel_first_kind()
        # approximate
        bessel_func_result = torch.exp(self.kappa) / torch.sqrt(2 * torch.pi * self.kappa) * (1 + 1/(8*self.kappa) + 9/(128*self.kappa**2))
        norm_coef = torch.pow(self.kappa, self.p/2-1) / ( (np.power(2*np.pi, self.p/2)) * bessel_func_result)
        dot_prod = self.kappa * (x @ mu)
        return norm_coef * torch.exp(dot_prod)

    def kl_divergence_(self, x):
        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)
        log_bessel_func_result = self.kappa - torch.log(torch.sqrt(2 * torch.pi * self.kappa) * (1 + 1/(8*self.kappa) + 9/(128*self.kappa**2)))
        log_C_p = (self.p/2 - 1) * torch.log(self.kappa) - self.p/2 * torch.log(torch.tensor(2 * torch.pi)) - log_bessel_func_result
        dot_prod = self.kappa * (x @ mu)
        kl_divergence = - log_C_p - dot_prod.mean()
        return kl_divergence


    def kl_divergence(self, x):
        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)
        bessel_func_result = self.modified_bessel_first_kind()
        log_bessel_func_result = torch.log(bessel_func_result)
        log_C_p = (self.p/2 - 1) * torch.log(self.kappa) - self.p/2 * torch.log(torch.tensor(2 * torch.pi)) - log_bessel_func_result
        dot_prod = self.kappa * (x @ mu)
        kl_divergence = -log_C_p - dot_prod.mean()
        return kl_divergence

    def kl_divergence_stopvmf(self, x):
        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)
        mu = mu.detach()
        kappa = self.kappa.detach()
        # log_bessel_func_result = kappa - torch.log(torch.sqrt(2 * torch.pi * kappa) * (1 + 1/(8*kappa) + 9/(128*kappa**2)))
        bessel_func_result = self.modified_bessel_first_kind()
        log_bessel_func_result = torch.log(bessel_func_result)
        log_C_p = (self.p/2 - 1) * torch.log(kappa) - self.p/2 * torch.log(torch.tensor(2 * torch.pi)) - log_bessel_func_result
        dot_prod = kappa * (x @ mu)
        kl_divergence = - log_C_p - dot_prod.mean()
        return kl_divergence

    def kl_divergence_stopx(self, x_):
        x = x_.detach()
        mu = self.mu / self.mu.norm(dim=-1, keepdim=True)
        log_bessel_func_result = self.kappa - torch.log(torch.sqrt(2 * torch.pi * self.kappa) * (1 + 1/(8*self.kappa) + 9/(128*self.kappa**2)))
        log_C_p = (self.p/2 - 1) * torch.log(self.kappa) - self.p/2 * torch.log(torch.tensor(2 * torch.pi)) - log_bessel_func_result
        dot_prod = self.kappa * (x @ mu)
        kl_divergence = - log_C_p - dot_prod.mean()
        return kl_divergence

class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images

        self.sup_class_num = 17 # ad hoc
        self.vmf_distributions = nn.ModuleList([VMFDistribution(64) for i in range(self.sup_class_num)])

        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.sup_criterion = SupConLoss()
        # self.ce_criterion = nn.CrossEntropyLoss()
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        # vmf_params_without_mu = [param for name, param in self.vmf_distributions.named_parameters() if ".mu" not in name]
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    # def on_after_backward(self):
    #     pass
        # print("kappa", self.vmf_distributions[0].kappa.item(),)
        # print("kappa.grad", self.vmf_distributions[0].kappa.grad, self.vmf_distributions[0].kappa.item())
        # print("mu.grad", self.vmf_distributions[0].mu.grad)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        # output_points = self.model_points(sparse_input).F
        r = self.model_points(sparse_input)
        output_points, output_points_class_level = r[0].F, r[1].F        
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images, output_images_class_level = self.model_images(batch["input_I"])

        del batch["sinput_F"]
        # del batch["sinput_C"]
        del sparse_input

        pc_feas = output_points_class_level[batch["pairing_points"]]

        losses = [
            # getattr(self, loss)(batch, output_points, output_images)
            # for loss in self.losses
            self.loss(batch, output_points, output_images),
            self.loss_supervise(batch, output_points_class_level, output_images_class_level),
        ]
        loss = torch.mean(torch.stack(losses))

        kl_loss = 0
        decay = 0.0001
        kl_loss_list = []
        for c_idx in range(self.sup_class_num):
            c_mask = batch["pairing_labels"] == c_idx

            if self.global_step <= 1 / decay:
                if c_mask.sum() != 0:
                    self.vmf_distributions[c_idx].z_mean.data = (1 - decay) * self.vmf_distributions[c_idx].z_mean.data +  decay * pc_feas[c_mask].mean(0).clone()
                dist.all_reduce(self.vmf_distributions[c_idx].z_mean.data, op=dist.ReduceOp.SUM)
                self.vmf_distributions[c_idx].z_mean.data /= dist.get_world_size()
                z_mean_norm = self.vmf_distributions[c_idx].z_mean.data.norm()
                self.vmf_distributions[c_idx].mu.data = self.vmf_distributions[c_idx].z_mean.data / z_mean_norm
                self.vmf_distributions[c_idx].kappa.data = z_mean_norm * (self.vmf_distributions[c_idx].p - z_mean_norm ** 2) / (1 - z_mean_norm ** 2)
            else:
                if c_mask.sum() != 0:
                    kl_loss_c = self.vmf_distributions[c_idx].kl_divergence_stopvmf(pc_feas[c_mask])
                    kl_loss_list.append(kl_loss_c)

        if self.global_step > 1 / decay:
            kl_loss = torch.mean(torch.stack(kl_loss_list))

        self.log("cl_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("kappa_0", self.vmf_distributions[0].kappa.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("kl_loss", kl_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        loss = loss + kl_loss
        
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_supervise(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        pairing_labels = batch["pairing_labels"]
        # idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)

        rho = batch['sinput_C'][pairing_points, 1]
        std_dev = rho.float().std()
        unique_samples, inverse_indices = torch.unique(rho, sorted=False, return_inverse=True)
        bandwidth = 1.06 * std_dev * (len(rho) ** (-1 / 5))
        kde_estimate = kde(unique_samples, rho, bandwidth=bandwidth)
        density = kde_estimate[inverse_indices]

        unique_elements, inverse_indices = torch.unique(pairing_labels, return_inverse=True)
        counts = torch.bincount(inverse_indices)  # Count occurrences of each unique element
        occurrence_array = counts[inverse_indices]  # Map counts back to the original array structure
    
        probabilities = 1 / density / occurrence_array
        idx = torch.multinomial(probabilities, num_samples=self.num_matches, replacement=False)

        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        s_labels = pairing_labels[idx]
        return self.sup_criterion(k, q, s_labels)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        # output_points = self.model_points(sparse_input).F
        r = self.model_points(sparse_input)
        output_points, output_points_class_level = r[0].F, r[1].F        
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images, output_images_class_level = self.model_images(batch["input_I"])        

        losses = [
            # getattr(self, loss)(batch, output_points, output_images)
            # for loss in self.losses
            self.loss(batch, output_points, output_images),
            self.loss_supervise(batch, output_points_class_level, output_images_class_level),
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                # "model_points": self.model_points.state_dict(),
                # "model_images": self.model_images.state_dict(),
                "state_dict": self.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
