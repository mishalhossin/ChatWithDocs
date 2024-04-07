import yaml


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

base_url = config.get('BASE_URL')
embeddings_model = config.get('EMBEDDINGS_MODEL')
chat_model = config.get('CHAT_MODEL')