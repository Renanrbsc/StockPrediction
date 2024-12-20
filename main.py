import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from controller.controller import MarketController

if __name__ == "__main__":
    controller = MarketController()
    controller.execute()
