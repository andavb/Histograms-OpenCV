from tkinter import *
from PIL import ImageTk, Image
from cv2 import *
import PIL.Image, PIL.ImageTk, PIL.ImageDraw
from tkinter.colorchooser import askcolor
from tkinter.filedialog import askopenfilename
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

img=""
filename=""
height=0
width=0
vrednosti = np.arange(0,256)

def odprisliko():
    global filename, img, height, width
    filename = askopenfilename()
    img = cv2.imread(filename)
    cvImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    height = np.size(cvImg, 0)
    width = np.size(cvImg, 1)


def rgbdisplay():
    global img, vrednosti
    img = cv2.imread("nova.png")
    cv2.imshow('image', img)

    b = img.copy()


    height = np.size(img, 0)
    width = np.size(img, 1)
    pixelsR = np.zeros((256), np.uint64)
    pixelsG = np.zeros((256), np.uint64)
    pixelsB = np.zeros((256), np.uint64)

    for x in range(0, height):
        for y in range(0, width):
            pixelsB[b[x, y, 0]] += 1
            pixelsG[b[x, y, 1]] += 1
            pixelsR[b[x, y, 2]] += 1

    plt.subplot(3, 1, 1)
    plt.plot(pixelsR, color="red")
    plt.title('R', fontsize="10")
    plt.subplot(3, 1, 2)
    plt.plot(pixelsG, color="green")
    plt.title('G', fontsize="10")
    plt.subplot(3, 1, 3)
    plt.plot(pixelsB, color="blue")
    plt.title('B', fontsize="10")
    plt.xlim([0, 256])
    plt.show()

def YCrCbdisplay():
    global img, vrednosti
    img = cv2.imread("nova.png")

    b = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    cv2.imshow('image', b)

    height = np.size(img, 0)
    width = np.size(img, 1)
    pixelsY = np.zeros((256), np.uint64)
    pixelsCB = np.zeros((256), np.uint64)
    pixelsCR = np.zeros((256), np.uint64)

    for x in range(0, height):
        for y in range(0, width):
            pixelsY[b[x, y, 0]] += 1
            pixelsCB[b[x, y, 1]] += 1
            pixelsCR[b[x, y, 2]] += 1

    plt.subplot(3, 1, 1)
    plt.plot(pixelsY, color="purple")
    plt.title('Y', fontsize="10")
    plt.subplot(3, 1, 2)
    plt.plot(pixelsCB, color="blue")
    plt.title('CB', fontsize="10")
    plt.subplot(3, 1, 3)
    plt.plot(pixelsCR, color="red")
    plt.title('CR', fontsize="10")
    plt.xlim([0, 256])
    plt.show()


def HSVbdisplay():
    global img, vrednosti
    img = cv2.imread("nova.png")

    b = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    cv2.imshow('image', b)

    height = np.size(img, 0)
    width = np.size(img, 1)
    pixelsH = np.zeros((256), np.uint64)
    pixelsS = np.zeros((256), np.uint64)
    pixelsV = np.zeros((256), np.uint64)

    for x in range(0, height):
        for y in range(0, width):
            pixelsH[b[x, y, 0]] += 1
            pixelsS[b[x, y, 1]] += 1
            pixelsV[b[x, y, 2]] += 1

    plt.subplot(3, 1, 1)
    plt.plot(pixelsH, color="red")
    plt.title('H', fontsize="10")
    plt.subplot(3, 1, 2)
    plt.plot(pixelsS, color="green")
    plt.title('S', fontsize="10")
    plt.subplot(3, 1, 3)
    plt.plot(pixelsV, color="blue")
    plt.title('V', fontsize="10")
    plt.xlim([0, 256])
    plt.show()

def GrayHist(ime):
    global img, vrednosti
    img = cv2.imread(ime)

    h = np.size(img, 0)
    w = np.size(img, 1)

    pixels = np.zeros((256), np.uint64)

    for x in range(0, h):
        for y in range(0, w):
            pixels[img[x, y, 0]] += 1

    plt.plot(pixels, color="gray")
    plt.title('gray', fontsize="10")
    plt.xlim([0, 256])
    plt.show()


def izenacitev():
    global height, width, img

    img = cv2.imread("nova.png")
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('gray.png', grayImg)

    height = np.size(img, 0)
    width = np.size(img, 1)

    GrayHist("gray.png")

    histogram = np.zeros((256), np.uint64)

    for x in range(0, height):
        for y in range(0, width):
            histogram[grayImg[x, y]] += 1

    komulativnaHistogram = np.zeros((256), np.uint64)
    komulativnaHistogram[0] = histogram[0]

    for p in range(1, 256):
        komulativnaHistogram[p] = komulativnaHistogram[p - 1] + histogram[p]

    Tp = np.zeros((256), np.uint64)
    for p in range(0, 256):
        Tp[p] = np.rint(((255) * komulativnaHistogram[p]) / (height * width)) #zaokorz

    koncnaSlika = np.zeros((height, width))
    for x in range(0, height):
        for y in range(0, width):
            koncnaSlika[x, y] = Tp[grayImg[x, y]]

    cv2.imwrite("gray2.png", koncnaSlika)

    GrayHist("gray2.png")

def izberi():
    var = input("RGB = 1,  YCrCb = 2, HSV = 3, izenacitev = 4 :")
    if var == "1":
        rgbdisplay()
    elif var == "2":
        YCrCbdisplay()
    elif var == "3":
        HSVbdisplay()
    elif var == "4":
        izenacitev()


izberi()