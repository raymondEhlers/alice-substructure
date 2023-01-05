"""Attempts to redirect stdout from c/c++ (ie. ROOT) to a logger.

NOTE: None of these work :-(

"""

import sys
import logging
from contextlib import redirect_stdout

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
        print(f"my_msg: {msg}. end")
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

from io import BytesIO, StringIO

# See: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile

# This didn't immediately work on macOS. I didn't dig further..
libc = ctypes.CDLL(None)
#c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        #libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
            )

    logger = logging.getLogger(__name__)
    #sys.stdout = StreamToLogger(logger, logging.INFO)
    #sys.stderr = StreamToLogger(logger, logging.ERROR)
    #log_stdout = StreamToLogger(logger, logging.INFO)
    #log_stderr = StreamToLogger(logger, logging.ERROR)
    # To access the original stdout/stderr, use sys.__stdout__/sys.__stderr__
    #sys.stdout = LoggerWriter(logger.info)
    #sys.stderr = LoggerWriter(logger.error)
    log_stdout = LoggerWriter(logger.info)
    log_stderr = LoggerWriter(logger.error)

    logger.info("Test from logger")
    print('Test to standard out')
    #raise Exception('Test to standard error')

    import ROOT
    print("Imported ROOT")
    h = ROOT.TH1D("test", "test", 10, 0, 1)
    #with pipes(stdout=log_stdout, stderr=log_stderr):
    #with sys_pipes():
    b = BytesIO()
    #with stdout_redirected(to=b):
    #with redirect_stdout(log_stdout):
    with stdout_redirector(b):
        h.Print()
    logger.info(b.getvalue().decode('utf-8'))
    print("done")

    f = io.BytesIO()
    with stdout_redirector(f):
        print('foobar')
        print(12)
        #libc.puts(b'this comes from C')
        os.system('echo and this is from echo')
    print("fully done")

    #import IPython; IPython.embed()
