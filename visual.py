# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 14:24 
# @Author : wzy 
# @File : visual.py
# ---------------------------------------
from matplotlib import pyplot as plt


def draw(teacher_history, student_history, epochs):

    x = list(range(1, epochs + 1))

    plt.subplot(2, 1, 1)
    plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='SwinOKD(teacher)')
    plt.plot(x, [student_history[i][0] for i in range(epochs)], label='SwinOKD(student)')

    plt.title('Test accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label='SwinOKD(teacher)')
    plt.plot(x, [student_history[i][1] for i in range(epochs)], label='SwinOKD(student)')

    plt.title('Test loss')
    plt.legend()

    plt.savefig("visual.png")
