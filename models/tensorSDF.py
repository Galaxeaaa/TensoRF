from .tensoRF import *

def my_grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class TensorSDF(TensorVMSplit):
    def compute_sdfeature(self, xyz_sampled):
        """
        Exactly the same content as compute_densityfeature. The difference is that the
        meaning of the output is the signed distance feature instead.
        """
        # plane + line basis
        coordinate_plane = (
            torch.stack(
                (
                    xyz_sampled[..., self.matMode[0]],
                    xyz_sampled[..., self.matMode[1]],
                    xyz_sampled[..., self.matMode[2]],
                )
            )
            # .detach()
            .view(3, -1, 1, 2)
        )
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]],
                xyz_sampled[..., self.vecMode[1]],
                xyz_sampled[..., self.vecMode[2]],
            )
        )
        coordinate_line = (
            torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1)
            # .detach()
            .view(3, -1, 1, 2)
        )

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            # plane_coef_point = F.grid_sample(
            plane_coef_point = my_grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_plane]],
                # align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            # line_coef_point = F.grid_sample(
            line_coef_point = my_grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_plane]],
                # align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )

        return sigma_feature

    def sdfeature2value(self, sd_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(sd_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(sd_features)

    def compute_sd_value(self, xyz_sampled):
        sd_feature = self.compute_sdfeature(xyz_sampled)
        return self.sdfeature2value(sd_feature)

    def sd2alpha(self, sd):
        # sigma, dist  [N_rays, N_samples]
        # print(f"sd shape: {sd.shape}")
        sd_shifted = torch.cat((sd[..., 1:], torch.zeros_like(sd[..., :1])), dim=-1)
        alpha = torch.relu(
            (torch.sigmoid(sd) - torch.sigmoid(sd_shifted)) / torch.sigmoid(sd)
        )

        T = torch.cumprod(
            torch.cat(
                [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )

        weights = alpha * T[:, :-1]  # [N_rays, N_samples]
        return alpha, weights, T[:, -1:]

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            print("alpha mask is not none")
            print(f"alpha mask shape: {self.alphaMask.alpha_volume.shape}")
            print(f"alpha mask: {self.alphaMask.alpha_volume}")
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            print("alpha mask is none. Use all ones.")
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sd = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sd_feature = self.compute_sdfeature(xyz_sampled)
            validsd = self.sdfeature2value(sd_feature)
            print(f"validsd.shape: {validsd.shape}")
            sd[alpha_mask] = validsd

        sd_shifted = torch.cat((sd[..., 1:], torch.zeros_like(sd[..., :1])), dim=-1)
        alpha = torch.relu(
            (torch.sigmoid(sd) - torch.sigmoid(sd_shifted)) / torch.sigmoid(sd)
        )

        return alpha

    def gradient(self, x):
        x.requires_grad_(True)
        # print(x.size())
        # print(x.requires_grad)
        y = self.compute_sd_value(x)
        # print(y.size())
        # print(y.requires_grad)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def eikonal_loss(self, sd_values, xyz_sampled):
        n = xyz_sampled.size()[0]
        gradients = self.gradient(xyz_sampled).squeeze()
        # print(f"gradients.shape = {gradients.size()}")
        
        loss = torch.sum((torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2) / n

        return loss

    def forward(
        self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1
    ):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples
            )
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])),
                dim=-1,
            )
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= ~alpha_mask
            ray_valid = ~ray_invalid

        sd = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            xyz_sampled.requires_grad_(True)
            # print("xyz_sampled: ", xyz_sampled.requires_grad)
            # sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            sd_feature = self.compute_sdfeature(xyz_sampled[ray_valid])
            # print("sd_feature: ", sd_feature.requires_grad)

            # validsigma = self.feature2density(sigma_feature)
            validsd = self.sdfeature2value(sd_feature)
            # print("validsd: ", validsd.requires_grad)
            sd[ray_valid] = validsd

            # d_output = torch.ones_like(sd[ray_valid], requires_grad=False, device=self.device)
            # gradients = torch.autograd.grad(
            #     outputs=sd[ray_valid],
            #     inputs=xyz_sampled[ray_valid],
            #     grad_outputs=d_output,
            #     create_graph=True,
            #     retain_graph=True,
            #     only_inputs=True)[0]

            # print(gradients.size())

        alpha, weight, bg_weight = self.sd2alpha(sd)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(
                xyz_sampled[app_mask], viewdirs[app_mask], app_features
            )
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map, sd, xyz_sampled[ray_valid]  # rgb, sigma, alpha, weight, bg_weight
