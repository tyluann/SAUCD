# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import config; cfg = config.cfg


OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING

class Logger():
    def __init__(self, log_file):
        # set log
        self._logger = logging.getLogger('')
        self._logger.setLevel(logging.INFO)
        # log_file = os.path.join(log_dir, log_name)
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        while len(self._logger.handlers) > 0:
            h = self._logger.handlers[0]
            self._logger.removeHandler(h)
        file_log = logging.FileHandler(log_file, mode='a')
        file_log.setLevel(logging.INFO)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%y-%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        console_log.setFormatter(formatter)
        # self._logger.handlers.clear()

        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)

log = None

def init_logger():
    log_file = os.path.join(cfg.dir_output_log, cfg.name + ".log")
    logger = Logger(log_file)
    return logger


def join_print(*args):
    return ' '.join([str(i) for i in args])

def printd(*args):
    log.debug(join_print(*args))
    
def printi(*args):
    log.info(join_print(*args))
    
def printw(*args):
    log.warning(join_print(*args))
    
def printc(*args):
    log.critical(join_print(*args))
    
def printe(*args):
    log.error(join_print(*args))