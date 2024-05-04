import argparse
import ollama

def download_model(model_name):
    ollama.pull(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download a model with ollama')
    parser.add_argument('--model_name', type=str, required=True, help='The name of the model to download')

    args = parser.parse_args()

    download_model(args.model_name)