"""
use SMPL triplane rendering as additional feature

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os
sys.path.append(os.getcwd())
import torch
from recon.gen.generator import Generator


class GeneratorTriplane(Generator):
    def prep_query_input(self, batch):
        """
        in addition to the crop center, send also the body center to query
        which is used in the triplane projection
        Args:
            batch: batch data from data loader

        Returns: query data dict

        """
        crop_center = batch.get('crop_center').to(self.device)  # (B, 3)
        body_center = batch.get('body_center').to(self.device)  # (B, 3)
        return {
            'crop_center': crop_center,
            "body_center": body_center
        }

    def get_grid_samples(self, sample_num, batch_size=1, body_center=None):
        """
        samples in a 3d grid around the human body center
        Args:
            sample_num:
            batch_size:
            z_0:
            body_center: (B, 3)

        Returns:

        """
        assert body_center is not None

        samples = torch.rand(batch_size, sample_num, 3).float().to(self.device)  # generate random sample points

        samples[:, :, 0] = samples[:, :, 0] * 2 - 1  # x: -1, 1
        samples[:, :, 1] = samples[:, :, 1] * 3 - 1.5  # y: -1.5, 1.5
        samples[:, :, 2] = samples[:, :, 2] * 1.2 - 0.6  # z: -0.6, 0.6

        # move the cube to human body center
        samples = samples + body_center.unsqueeze(1).to(self.device)

        return samples


