from logging import config
from logging.handlers import RotatingFileHandler

import codecs
import logging
import os
import yaml

base_dir = os.path.dirname(os.path.dirname(__file__))
level = {
    "local": logging.INFO,
    "cloud": logging.INFO,
    "local_tiny": logging.INFO,
}


class Logging(object):
    instance = None

    @staticmethod
    def logger(name):
        if Logging.instance is None:
            print('init logger instance ...')
            with codecs.open('logs/logging.yaml', 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r, Loader=yaml.FullLoader))
            Logging.instance = logging
        logger_ = Logging.instance.getLogger(name)
        return logger_


# Logger for api
def logger(name, now_level):
    logger_ = Logging.logger(name)

    # Add handler
    f_path = "logs/model.log"
    handler = RotatingFileHandler(f_path, maxBytes=10485760, backupCount=20, encoding='utf8')
    handler.setLevel(level[now_level])
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [line:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    logger_.addHandler(handler)

    return logger_
