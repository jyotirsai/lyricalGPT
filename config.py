
def get_config():
    return {
        'data_file_path': '/content/drive/My Drive/Notebooks/lyricalGPT/spotify_millsongdata.csv',
        'context_size': 8, # must be > min_token_length and < max_token_length
        'min_token_length': 100,
        'max_token_length': 500,
        'batch_size': 8,
        'epochs': 10,
        'tokenizer_file': 'tokenizer_{0}.json',
    }