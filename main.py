from multiprocessing import cpu_count
from pathlib import Path
from pydub import AudioSegment
import torch
from fairseq import checkpoint_utils

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from audio import load_audio
from pipeline import Pipeline as VC
import requests,os
from zipfile import ZipFile


class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                print("16 series/10 series P40 forced single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open("configs" / config_file, "r") as f:
                        strr = f.read().replace("true", "false")
                    with open("configs" / config_file, "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='', )
    hubert = models[0]
    hubert = hubert.to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert


def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu')
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


def svc_infer(index_path, index_rate, input_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model):
    if not output_path.endswith(".mp3"):
        Print('.mp3 only !')
        return
    audio = load_audio(input_path, 16000)
    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    audio_opt = vc.pipeline(hubert_model, net_g, 0, audio, input_path, times, pitch_change, f0_method, index_path, index_rate, if_f0, filter_radius, tgt_sr, 0, rms_mix_rate, version, protect, crepe_hop_length)
    print('Succesfully converted!\nSaving the mp3 file...')
    AudioSegment(audio_opt.tobytes(), frame_rate=tgt_sr, sample_width=audio_opt.dtype.itemsize, channels=1 if len(audio_opt.shape) == 1 else audio_opt.shape[1]).export(output_path, format='mp3')
    print(f'saved in {output_path}')
    

def get_model_path(folder):
    svc_model_path, svc_index_path = None,None
    for filee in os.listdir(folder):
        if filee.endswith('.index'):
            svc_index_path = folder+'/'+filee
        if filee.endswith('.pth'):
            svc_model_path = folder+'/'+filee
    return svc_model_path, svc_index_path

def extract(filename,output_path=""):
    with ZipFile(filename,'r') as zObject:
        zObject.extractall(path=output_path)

def down(url,output_file):
    r = requests.get(url)
    with open(output_file,'wb') as f:
        f.write(r.content)
