"""Shared host-user ownership for recorder-created files, including locks."""
# Snapshot of the canonical collector's aloha/output_permissions.py on 103.
import os
import stat


def output_owner():
    defaults = (1000, 1000) if os.geteuid() == 0 else (os.geteuid(), os.getegid())
    values = tuple(int(os.environ.get(key) or default) for key, default in
                   zip(('ALOHA_OUTPUT_UID', 'ALOHA_OUTPUT_GID'), defaults))
    if min(values) < 0:
        raise ValueError('output owner IDs must be nonnegative')
    return values


def normalize_fd(fd, *, uid=None, gid=None):
    current = os.fstat(fd)
    if stat.S_ISDIR(current.st_mode):
        mode = 0o775
    elif stat.S_ISREG(current.st_mode):
        if current.st_nlink != 1:
            raise ValueError('refusing multiply linked output file')
        mode = 0o664
    else:
        raise ValueError('unsupported output file type')
    owner = output_owner() if uid is None or gid is None else (uid, gid)
    if (current.st_uid, current.st_gid) != owner:
        os.fchown(fd, *owner)
    os.fchmod(fd, mode)


def normalize_path(path, *, uid=None, gid=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        normalize_fd(fd, uid=uid, gid=gid)
    finally:
        os.close(fd)


def open_recording_lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o664)
    try:
        normalize_fd(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise
