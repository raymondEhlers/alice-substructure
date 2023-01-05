"""Attempts to redirect stdout from c/c++ (ie. ROOT) to a logger.

NOTE: None of these work :-(

"""

import sys
import logging

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.removesuffix('\n'))
            self.logfct(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass


from wurlitzer import pipes, sys_pipes

import os
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        #with open(to, 'w') as file:
        _redirect_stdout(to=to)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

from io import BytesIO

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            )

    logger = logging.getLogger(__name__)
    #sys.stdout = StreamToLogger(log, logging.INFO)
    #sys.stderr = StreamToLogger(log, logging.ERROR)
    log_stdout = StreamToLogger(logger, logging.INFO)
    log_stderr = StreamToLogger(logger, logging.ERROR)
    # To access the original stdout/stderr, use sys.__stdout__/sys.__stderr__
    #sys.stdout = LoggerWriter(logger.info)
    #sys.stderr = LoggerWriter(logger.error)
    #log_stdout = LoggerWriter(logger.info)
    #log_stderr = LoggerWriter(logger.error)

    logger.info("Test from logger")
    print('Test to standard out')
    #raise Exception('Test to standard error')

    import ROOT
    print("Imported ROOT")
    h = ROOT.TH1D("test", "test", 10, 0, 1)
    #with pipes(stdout=log_stdout, stderr=log_stderr):
    #with sys_pipes():
    b = BytesIO()
    with stdout_redirected(to=b):
        h.Print()
    print(b.decode())

    #import IPython; IPython.embed()
