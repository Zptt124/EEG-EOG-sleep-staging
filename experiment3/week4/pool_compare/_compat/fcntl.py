"""fcntl 兼容垫片(仅 Windows 本机使用):C 的脚本导入 fcntl 只为 run() 里的排他锁,
该路径在第四周对照中不会走到。提供同名的空实现,不改动 C 的任何源文件。"""

LOCK_EX = 2
LOCK_NB = 4
LOCK_SH = 1
LOCK_UN = 8


def flock(fd, operation):
    return None


def lockf(fd, command, length=0, start=0, whence=0):
    return None
