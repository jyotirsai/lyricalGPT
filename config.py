
def get_config():
    return {
        'data_file_path': '/content/drive/My Drive/Notebooks/lyricalGPT/spotify_millsongdata.csv',
        'context_size': 64, # must be < max_token_length
        'min_token_length': 100,
        'max_token_length': 500,
        'batch_size': 8,
        'epochs': 10,
        'n_heads': 8,
        'n_layers': 6,
        'embed_size': 512,
        'ff_size': 2048,
        'dropout': 0.1,
        'lr': 3e-4,
        'preload': None,
        'model_basename': 'lyricalGPT',
        'tokenizer_file': 'tokenizer_{0}.json',
    }