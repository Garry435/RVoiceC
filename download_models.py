import requests

DOWNLOAD_LINK = 'https://huggingface.co/Garry908/sample-test/resolve/main/'
def dl_model(link, model_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=16384):
                f.write(chunk)

model_names = ['hubert_base.pt', 'rmvpe.pt']
for model in model_names:
    print(f'Downloading {model}...')
    dl_model(DOWNLOAD_LINK, model)
print('All models downloaded!')
