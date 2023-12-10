from .tensoRF import *


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
            .detach()
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
            .detach()
            .view(3, -1, 1, 2)
        )

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(
                self.density_plane[idx_plane],
                coordinate_plane[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(
                self.density_line[idx_plane],
                coordinate_line[[idx_plane]],
                align_corners=True,
            ).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )

        return sigma_feature

    def sd2alpha(sd, dist):
        # sigma, dist  [N_rays, N_samples]
        sd_shifted = torch.cat(sd[1:], 0)
        alpha = torch.sigmoid((sd - sd_shifted) / sd)

        T = torch.cumprod(
            torch.cat(
                [torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )

        weights = alpha * T[:, :-1]  # [N_rays, N_samples]
        return alpha, weights, T[:, -1:]

    def sdfeature2value(self, sd_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(sd_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(sd_features)

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

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            # sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            sd_feature = self.compute_sdfeature(xyz_sampled[ray_valid])

            # validsigma = self.feature2density(sigma_feature)
            validsigma = self.sdfeature2weight(sd_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

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

        return rgb_map, depth_map  # rgb, sigma, alpha, weight, bg_weight
