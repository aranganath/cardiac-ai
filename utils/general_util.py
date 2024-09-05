# Logging

import logging
import os, os.path
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('fast')
log.setLevel(logging.DEBUG)
log.handlers = []       # no duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)
logging.Logger.infov = _infov


# Etc

import getpass
import tempfile

def get_temp_dir():
    user = getpass.getuser()

    for t in ('/p/lustre1/' + user,
              '/usr/workspace/' + user,
              '/data/non-ssi/' + user,
              tempfile.gettempdir()):
        if os.path.exists(t):
            return mkdir_p(t + '/cardiac-ai.tmp')
    return None

def mkdir_p(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass
    return path
