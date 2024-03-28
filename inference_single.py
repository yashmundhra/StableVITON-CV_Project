import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img

class _Module:
    pass
module = _Module()

def init():
    module.batch_size = 32
    module.img_H = 512
    module.img_W = 384

    module.config = OmegaConf.load("/home/jun/StableVITON/configs/VITON512.yaml")
    module.config.model.params.img_H = 512
    module.config.model.params.img_W = 384
    module.params = module.config.model.params

    module.model = create_model(config_path="/home/jun/StableVITON/configs/VITON512.yaml")
    module.model.load_state_dict(torch.load("/home/jun/StableVITON/checkpoints/VITONHD.ckpt", map_location="cpu"))
    module.model = module.model.cuda()
    module.model.eval()


@torch.no_grad()
def single_infer(args):
    sampler = PLMSSampler(module.model)
    dataset = getattr(import_module("dataset"), module.config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=module.img_H,
        img_W=module.img_W,
        is_paired=False,
        is_test=True,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=1, shuffle=False, batch_size=1, pin_memory=True)

    shape = (4, module.img_H//8, module.img_W//8) 
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for batch in dataloader:
        z, c = module.model.get_input(batch, module.params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = module.model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = module.model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = module.model.q_sample(z, ts)     

        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
        )

        x_samples = module.model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(save_dir, 'output.png')
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])