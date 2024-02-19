#from share import *

import os
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataset_aug import VITONHDDataset_aug

from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datetime import datetime
import yaml

#from API.slack import SlackMessenger

import socket

def get_ip_address():
    try:
        host_name = socket.gethostname()
        ip_address = socket.gethostbyname(host_name)
        return ip_address
    except:
        return "Unknown"

class EpochEndSlackNotifier(Callback):
    def __init__(self, messenger, notify_every_n_epochs=5):
        self.messenger = messenger
        self.notify_every_n_epochs = notify_every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            epoch = trainer.current_epoch
            learning_rate = pl_module.trainer.optimizers[0].param_groups[0]['lr']
            if (epoch + 1) % self.notify_every_n_epochs == 0:
                ip_address = get_ip_address()
                message = f"Server : {ip_address} \n\nEpoch {epoch + 1} checkpoints saved ! \n\n Learning rate : {learning_rate}"
                self.messenger.send_message(message)

if __name__ == '__main__':     
    current_time = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')   
    
    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{current_time}/',
        filename='model-{epoch:02d}',
        every_n_epochs=10,
        save_weights_only=False,
        save_top_k=-1        
    )
    
    trainer = pl.Trainer(
        gpus=1,
        #accelerator='ddp',
        precision=32,
        max_epochs=2000,
        callbacks=[checkpoint_callback]  
    )
        
    viton_config_path_model = './configs/VITON512.yaml'
    resume_path = '../stableviton_lightning/models/clone_with_control_model_20240204.ckpt'    
    batch_size = 10
    logger_freq = 300
    learning_rate = 1e-4
    epochs = 2000     
    
    if trainer.global_rank == 0:
        # 설정 저장 로직         
    
        control_model = True
        model_diffusion_model_warp_flow_blks = True
        model_diffusion_model_warp_zero_convs = True
        model_diffusion_model_input_clocks = True
        first_stage_model_all = False            
        proj_out = True
        lastzc = True    
        
        cond_stage_model_final_ln = False
        cond_stage_model_mapper_resblocks = False
        
        learnable_vector = False
        data_augmentation = True

        checkpoint_dir = f'checkpoints/{current_time}/'       
        os.makedirs(checkpoint_dir, exist_ok=True) 
        
        config = {
            'checkpoint_dir': checkpoint_dir,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'config_path': viton_config_path_model,
            'resume_path' : resume_path,
            'control_model' : control_model,                
            'model_diffusion_model_warp_flow_blks' : model_diffusion_model_warp_flow_blks,
            'model_diffusion_model_warp_zero_convs' : model_diffusion_model_warp_zero_convs,
            'model_diffusion_model_input_clocks' : model_diffusion_model_input_clocks,
            'learnable_vector' : learnable_vector,
            'proj_out' : proj_out,
            'lastzc' : lastzc,
            'first_stage_model_decoder' : first_stage_model_all,
            'cond_stage_model_final_ln' : cond_stage_model_final_ln,
            'cond_stage_model_mapper_resblocks' : cond_stage_model_mapper_resblocks,
            'data_augmentation' : data_augmentation
        }
        
        with open(viton_config_path_model, 'r') as file:
            viton_config = yaml.load(file, Loader=yaml.FullLoader)

        # YAML 파일을 checkpoint_dir에 저장
        viton_config_file = os.path.join(checkpoint_dir, 'VITON512.yaml')
        with open(viton_config_file, 'w') as file:
            yaml.dump(viton_config, file, sort_keys=False)
        
        config_file = os.path.join(checkpoint_dir, 'config.yaml')        

        # YAML 파일로 설정 저장
        with open(config_file, 'w') as file:
            yaml.dump(config, file, sort_keys=False)
            
        print('Configs !')
        print(config)
    
    
    model = create_model(viton_config_path_model).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    #model.only_mid_control = only_mid_control

    # Misc
    dataset = VITONHDDataset_aug(data_root_dir='../stableviton_lightning/datasets',img_H=512, img_W=384, is_paired=True, is_test=False, is_sorted=False)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    
    trainer.fit(model, dataloader)
    
