import shutil
import os
import random


def move_ext(src, dst, ext):
    """
    Move file based on its extension
    :param src: file source path
    :param dst: file destination path
    :param ext: extension of the file to be moved
    :return:
    """
    for file in os.listdir(src):
        if file.endswith(ext):
            shutil.move(os.path.join(src, file), os.path.join(dst, file))


def move_n(src, dst, n, sort=True, rndm=False):
    """
    Move n number of file
    :param src: file source path
    :param dst: file destination path
    :param n: number of file to be moved
    :param sort: sort file name if True
    :param rndm: random file name if True
    :return:
    """
    if sort and rndm:
        raise ValueError('sort and rndm cannot be both True')
    fn = []
    for file in os.listdir(src):
        fn.append(file)
    if sort:
        fn.sort()
    if rndm:
        random.shuffle(fn)
    for file in fn[:n]:
        shutil.move(os.path.join(src, file), os.path.join(dst, file))


def move_list(src, dst, fn_list, ext):
    """
    Move file based on list of name
    :param src: file source path
    :param dst: file destination path
    :param fn_list: list of file name
    :param ext: file extension on destination (None if src ext == dst ext)
    :return:
    """
    if ext is None:
        for file in fn_list:
            shutil.move(os.path.join(src, file), os.path.join(dst, file))
    else:
        for file in fn_list:
            shutil.move(os.path.join(src, os.path.splitext(file)[0]+'.'+ext),
                        os.path.join(dst, os.path.splitext(file)[0]+'.'+ext))