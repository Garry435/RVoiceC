import requests
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
def dl_model(link, model_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(f"mdxnet_models/{model_name}", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

dl_model(MDX_DOWNLOAD_LINK,"UVR-MDX-NET-Voc_FT.onnx")
