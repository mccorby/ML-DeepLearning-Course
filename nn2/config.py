from contextlib import contextmanager


@contextmanager
def open_config_if_exists(filename, mode='r'):
    try:
        f = open(filename, mode)
    except (IOError, err):
        yield None, err
    else:
        try:
            yield f, None
        finally:
            f.close()
