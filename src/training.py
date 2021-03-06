from src.utils.common import read_config
from src.utils.data_mgmt import get_data
import argparse

def training(config_path):
    config = read_config(config_path)
    validation_datasize = config['params']['validation_datasize']
    (x_train_full, y_train_full), (x_valid, y_valid), (x_test, y_test) = get_data()


if __name__ == '__main.py__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default = "config.yaml")
    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)