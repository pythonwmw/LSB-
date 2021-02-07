import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as pmg  # mpimg 用于读取图片
import numpy as np
import pylab as pl
import random
import threading
import multiprocessing
import itertools


# 对像素块进行Z字排序
def Z(tmp):
    size = 8
    a = [0] * 64
    n = 0
    i = 0
    j = 0
    status = 0  # 标记当前状态 0：右上运动 1：左下运动
    while (n < size * size):
        if (((i == 0 and j % 2 != 0) or (j == 1 and i == 0)) and j != size - 1):  # 当前位于像素矩阵的最左边的奇数位且不是边界，向下移动,下一步向右上运动
            a[n] = tmp[i][j]
            j = j + 1
            n = n + 1
            status = 0
        if (((i == 0 and j % 2 != 0) or (j == 1 and i == 0)) and j == size - 1):  # 当前位于像素矩阵的最左边的奇数位且是边界，向右移动,下一步向右上运动
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 0
        elif ((j == 0 and i % 2 == 0 and i > 1) or (i == 0 and j == 0)):  # 当前位于像素矩阵的最上面的偶数位，向右移动，下一步向左下运动
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 1
        elif ((i == 0 and j % 2 == 0 and j > 1) or (j == 0 and i == 0)):  # 当前位于像素矩阵的最左边的偶数位，向右上移动
            a[n] = tmp[i][j]
            j = j - 1
            i = i + 1
            n = n + 1
            status = 0
        elif ((j == 0 and i % 2 != 0) or (i == 1 and j == 0)):  # 当前位于像素矩阵的最上面的奇数位，向左下移动
            a[n] = tmp[i][j]
            i = i - 1
            j = j + 1
            n = n + 1
            status = 1
        elif ((i == size - 1 and j % 2 != 0)):  # 当前位于像素矩阵的最右边的奇数位，向下移动,下一步向左下运动
            a[n] = tmp[i][j]
            j = j + 1
            n = n + 1
            status = 1
        elif ((j == size - 1 and i % 2 == 0 and i > 1)):  # 当前位于像素矩阵的最下面的偶数位，向右移动，下一步向右上移动
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 0
        elif ((i == size - 1 and j % 2 == 0 and j > 1)):  # 当前位于像素矩阵的最右边的偶数位，向左下移动
            a[n] = tmp[i][j]
            i = i - 1
            j = j + 1
            n = n + 1
            status = 1
        elif ((j == size - 1 and i % 2 != 0)):  # 当前位于像素矩阵的最下面的奇数位，向右上移动
            a[n] = tmp[i][j]
            i = i + 1
            j = j - 1
            n = n + 1
            status = 0
        else:  # 不是边界条件时，使用状态值判断移动方向
            if (status == 0):  # 右上运动
                a[n] = tmp[i][j]
                i = i + 1
                j = j - 1
                n = n + 1
                status = 0
            elif (status == 1):  # 左下运动
                a[n] = tmp[i][j]
                i = i - 1
                j = j + 1
                n = n + 1
                status = 1
    return a


def LSB(fig, rate):
    # s = int(512 * 512 * rate)
    # sec = [0] * (s)
    # k = 0
    #
    # for i in range(s):
    #     sec[i] = random.randint(0, 1)
    #
    # #    print sec
    # #    print fig
    #
    # for i in range(512):
    #     for j in range(512):
    #         if (k < s):
    #             if (sec[k] == 1 and fig[i][j] % 2 == 0):  # 偶数嵌入1
    #                 fig[i][j] = fig[i][j] + 1
    #                 k += 1
    #             elif (sec[k] == 1 and fig[i][j] % 2 == 1):  # 奇数嵌入1
    #                 fig[i][j] = fig[i][j] + 0
    #                 k += 1
    #             elif (sec[k] == 0 and fig[i][j] % 2 == 0):  # 偶数嵌入0
    #                 fig[i][j] = fig[i][j] + 0
    #                 k += 1
    #             elif (sec[k] == 0 and fig[i][j] % 2 == 1):  # 奇数嵌入0
    #                 fig[i][j] = fig[i][j] - 1
    #                 k += 1
    return fig


# 计算像素相关性
def Calculate(a):
    res = 0
    for i in range(63):
        if (a[i + 1] > a[i]):
            res = res + a[i + 1] - a[i]
        else:
            res = res + a[i] - a[i + 1]
    return res


# 0翻转
def F0(val):
    return val


# 正翻转
def F1(val):
    if (val % 2 == 0 and val != 1):  # 偶数加一
        val = val + 1
    elif (val % 2 == 1 or val == 1):  # 奇数减一
        val = val - 1
    return val


# 负翻转
def F_1(val):
    if (val % 2 == 0 and val != 1):  # 偶数减一
        val = val - 1
    elif (val % 2 == 1 or val == 1):  # 奇数加一
        val = val + 1
    return val


# 生成随机数组
def Random(typ):
    ran = [0] * 64
    if (typ == 1):
        for i in range(64):
            ran[i] = random.randint(0, 1)
    elif (typ == -1):
        for i in range(64):
            ran[i] = random.randint(-1, 0)
    return ran


# RS隐写分析
def RS(tmp):
    # ==============================================================================
    #     非负反转
    # ==============================================================================
    ran = Random(1)
    rev = [[0 for co in range(8)] for ro in range(8)]  # 进行反转后的二维数组
    rm = 0
    sm = 0

    r1 = Z(tmp)
    res1 = Calculate(r1)  # 反转之前的像素相关性

    k = 0
    for i in range(8):
        for j in range(8):
            if (ran[k] == 0):  # F0翻转
                rev[i][j] = F0(tmp[i][j])
            elif (ran[k] == 1):  # F1翻转
                rev[i][j] = F1(tmp[i][j])
            k = k + 1

    r2 = Z(rev)  # 将图像块进行Z字形排序
    res2 = Calculate(r2)  # 翻转之后的像素相关性

    if (res1 > res2):
        sm = sm + 1
    elif (res1 < res2):
        rm = rm + 1

    # ==============================================================================
    #     非正翻转
    # ==============================================================================
    k = 0
    r_m = 0
    s_m = 0
    ran = Random(-1)
    rev = [[0 for co in range(8)] for ro in range(8)]  # 进行反转后的二维数组
    for i in range(8):
        for j in range(8):
            if (ran[k] == 0):
                rev[i][j] = F0(tmp[i][j])
            elif (ran[k] == -1):
                rev[i][j] = F_1(tmp[i][j])
            k = k + 1

    r3 = Z(rev)  # 将图像块进行Z字形排序
    res3 = Calculate(r3)  # 翻转之后的像素相关性

    if (res1 > res3):
        s_m = s_m + 1
    elif (res1 < res3):
        r_m = r_m + 1

    res = [rm, sm, r_m, s_m]
    return res


def Dblock(picture, flag=2):  # 图片分块 flag代表方块的边长
    rs_picture = picture.copy()
    if flag == 2:
        BlockSize = 4;
        BlockCol = 2;
        BlockRow = 2;
        x, y = rs_picture.shape[0], rs_picture.shape[1]
        bufsize = x * y
        Block_p = np.zeros((bufsize // BlockSize, BlockRow, BlockCol));
        Blocknum = 0
        for i in range(0, x // BlockRow):
            for j in range(0, y // BlockCol):
                Block_p[Blocknum, 0, 0] = rs_picture[BlockRow * i, BlockCol * j]
                Block_p[Blocknum, 0, 1] = rs_picture[BlockRow * i, BlockCol * j + 1]
                Block_p[Blocknum, 1, 0] = rs_picture[BlockRow * i + 1, BlockCol * j]
                Block_p[Blocknum, 1, 1] = rs_picture[BlockRow * i + 1, BlockCol * j + 1]
                Blocknum = Blocknum + 1;
        return Block_p
    elif flag == 8:
        BlockSize = 64;
        BlockCol = 8;
        BlockRow = 8;
        x, y = rs_picture.shape[0] // 8, rs_picture.shape[1] // 8
        Block_p = np.zeros((x * y, 8, 8))
        Blocknum = 0
        for i in range(x):
            for j in range(y):
                for k in range(8):
                    for h in range(8):
                        Block_p[Blocknum, k, h] = rs_picture[i * 8 + k, j * 8 + h]
                Blocknum += 1
        return Block_p


def anlysis(watermark):
    if True:
        compu = Dblock(watermark, 8)
        result = [0] * 4
        # 计算RM,SM等
        for i in range(compu.shape[0]):
            res = RS(compu[i, :, :])
            for n in range(4):
                result[n] = result[n] + res[n]
        # print('rm=', result[0], '  sm=', result[1], '  r_m=', result[2], '  s_m=', result[3])

        # 对加水印后的图像最低位取反
        watermark1 = watermark.copy()
        for i in range(watermark1.shape[0]):
            for j in range(watermark1.shape[1]):
                watermark1[i, j] = F1(watermark1[i, j])

        compu1 = Dblock(watermark1, 8)
        result1 = [0] * 4
        # 计算RM,SM等
        for i in range(compu1.shape[0]):
            res = RS(compu1[i, :, :])
            for n in range(4):
                result1[n] = result1[n] + res[n]
        print('rm=', result1[0], '  sm=', result1[1], '  r_m=', result1[2], '  s_m=', result1[3])
        # 计算隐写率
        d0 = result[0] - result[1]
        d1 = result1[0] - result1[1]
        d2 = result[2] - result[3]
        d3 = result1[2] - result1[3]
        args = [2 * (d1 + d0), d2 - d3 - d1 - 3 * d0, d0 - d2]
        a, b = np.roots(args)
        if a > b:
            print("估计隐写率为：", (b / (b - 0.5)).real)
            p = b / (b - 0.5)
        else:
            print("估计隐写率为：", (a / (a - 0.5)).real)
            p = a / (a - 0.5)
    return p.real




##-----main-----##
def RS2(pic_path,calculate_time,rate):
    result_dict = {}
    origin_rgb = pmg.imread(pic_path)
    # 取第一个通道的图像
    origin = np.array(origin_rgb[:, :, 0])
    all_piexl = origin.size
    watermark = LSB(origin, rate)
    pool = multiprocessing.Pool(3)
    rel = pool.map(anlysis,[watermark]*calculate_time)
    for index,value in zip(range(calculate_time),rel):
        result_dict[str(index+1)]=value
    return (result_dict,all_piexl)


