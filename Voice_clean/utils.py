from Voice_clean.mdx import run_mdx
import json,os
def preprocess_audio(orig_audio_path,mdxnet_models_dir='Voice_clean/mdxnet_models',audio_output_dir='Voice_clean/audio_outputs'):
    os.makedirs(f"Voice_clean/audio_outputs/",exist_ok=True)
    with open('Voice_clean/mdxnet_models/model_data.json') as infile:
        mdx_model_params = json.load(infile)
    print('[~] Cleaning Vocals')
    vocals_path = run_mdx(mdx_model_params, audio_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_audio_path, denoise=True,m_threads=6)
    return vocals_path
