"""
for the model that predicts visibility score

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os
sys.path.append(os.getcwd())
import torch
from recon.gen.generator_triplane import GeneratorTriplane


class GeneratorTriplaneVis(GeneratorTriplane):
    def get_out_names(self):
        "additional visibility predictor"
        return ['points', 'pca_axis', 'parts', 'centers', "visibility"]

    def compose_outdict(self, batch_size, out_dict, out_names, samples_count,
                        obj_mask=False, query_input=None):
        """query object poses here
        out_names = ['points', 'pca_axis', 'parts', 'centers', 'visibility']
        for each output, the shape is (B, ...)
        e.g. visibility: (B, 1), pca: (B, 3, 3), parts: (B, K)
        """
        for name in out_names:
            out_batch = out_dict[name]
            batch_comb = []
            for i in range(batch_size):
                if name == 'points':
                    out_i = torch.cat(out_batch[i], 0)  # points: (N, 3)
                    batch_comb.append(out_i[:samples_count, :])
                    continue
                out_i = torch.cat(out_batch[i], -1)
                out_i = out_i[..., :samples_count]

                # for object: only keep points projected to the object mask
                if obj_mask:
                    mask = out_dict['obj_mask'][i][:samples_count]
                else:
                    mask = torch.ones(samples_count, dtype=bool)

                if name == 'parts':
                    out_i = torch.argmax(out_i, 0)
                elif name == 'pca_axis':
                    # print(out_i.shape, mask.shape)
                    out_i = torch.mean(out_i[:, :, mask], -1)
                elif name in ['centers', 'visibility']:
                    out_i = torch.mean(out_i[:, mask], -1)  # compute the average centers

                batch_comb.append(out_i)
            out_dict[name] = torch.stack(batch_comb, 0)

        nan_values = torch.zeros_like(out_dict['centers']) + float('nan')  # backward compatibility
        out_dict['centers'] = torch.cat([nan_values, out_dict['centers']], 1) # (B, 6)
        return out_dict