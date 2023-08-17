import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tkinter import filedialog
import seaborn as sns; sns.set_theme()
import numpy
import cv2

def imgplot(figura: np.array):
      plt.figure(figsize=(20, 20))
      plt.imshow(figura, cmap='gray')
      plt.show()
arquivo = filedialog.askopenfilename(title='"pen')
figura = Image.open(arquivo)
figura = ImageOps.grayscale(figura)
figura = figura.resize(size=(36, 36))
largura, altura = figura.size
data = list(figura.getdata())
data = [data[offset:offset+largura] for offset in range(0, largura*altura, largura)]
array = np.array(data, dtype=np.uint8)
imgplot(figura)
sharpen = np.array([
         [0, -1, 0],
        [-1, 5, -1],
         [0, -1, 0]
])
blur = np.array([
        [0.0625, 0.125, 0.0625],
         [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625]
])
topsobel = np.array([
         [1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]
])
bottomsobel = np.array([
              [-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]
])
leftsobel = np.array([
           [1, 0, -1],
           [2, 1, -2],
           [1, 0, -1]
])
rightsobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
def doubleimgplot(img1: np.array, img2: np.array):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(img2, cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('Original')
    plt.show()
def calcularpixeltotal(tamfigura: int, kernelsize: int) -> int:
    pixels = 0
    for i in range(tamfigura):
        somador = i + kernelsize
        if somador <= tamfigura:
            pixels += 1
    return pixels
def convolucao(img: np.array, kernel: np.array) -> np.array:
    parametros = calcularpixeltotal(tamfigura=img.shape[0],kernelsize=kernel.shape[0])
    k = kernel.shape[0]
    ConvolucionaImage = np.zeros(shape=(parametros, parametros))
    for i in range(parametros):
        for j in range(parametros):
            matriz = img[i:i+k, j:j+k]
            ConvolucionaImage[i, j] = np.sum(np.multiply(matriz, kernel))
    return ConvolucionaImage
fig= plt.subplots()
sns.set(font_scale = 0.4)
img2=sns.heatmap(array, cmap='gray',annot=True, fmt="d",cbar=False,yticklabels=False,xticklabels=False)
plt.show()
for i in range(3):
  print("[")
  for j in range(3):
    print(array[i][j])
print("por qual tipo de matriz irá multiplicar sua imagem?")
print("a- Emboss")

print("b- Bottom Sobel")

print("c- Sharpen")

print("d- Blur")

print("e- Identity")

print("f- Top Sobel")

print("g- Outline")

print("h- Left Sobel")

print("i- Right Sobel")

print("j- escolher uma matriz")

escolha= input("qual seria sua matriz?:   ")
if (escolha=='a'):
    imgemboss = convolucao(np.array(figura), emboss)
    imgemboss
    print("Matriz de emboss")
    print(emboss)
    arrayemb = np.array(imgemboss, dtype=np.uint8)
    embval = sns.heatmap(arrayemb, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False, xticklabels=False)
    plt.show()
    doubleimgplot(figura, imgemboss)
elif (escolha=='b'):
    imagemBotS = convolucao(np.array(figura), bottomsobel)
    imagemBotS
    print("Matriz de bottom sobel")
    print(bottomsobel)
    arraybots = np.array(imagemBotS, dtype=np.uint8)
    botval = sns.heatmap(arraybots, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False, xticklabels=False)
    plt.show()
    doubleimgplot(figura, imagemBotS)
elif (escolha=='c'):
    imgsharped = convolucao(np.array(figura), sharpen)
    imgsharped
    print("Matriz de sharpen")
    print(sharpen)
    arrayshap = np.array(imgsharped, dtype=np.uint8)
    sharpval = sns.heatmap(arrayshap, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False,
                           xticklabels=False)
    plt.show()
    doubleimgplot(figura, imgsharped)
elif (escolha=='d'):
    imagemBlur = convolucao(np.array(figura), blur)
    imagemBlur
    print("Matriz de blur")
    print(blur)
    arrayblur = np.array(imagemBlur, dtype=np.uint8)
    valordoblur = sns.heatmap(arrayblur, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False,
                              xticklabels=False)
    plt.show()
    doubleimgplot(figura, imagemBlur)
elif (escolha=='e'):
    imgidentidade = convolucao(np.array(figura), identity)
    imgidentidade
    print("Matriz de identity")
    print(identity)
    arrayident = np.array(imgidentidade, dtype=np.uint8)
    idenval = sns.heatmap(arrayident, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False,
                          xticklabels=False)
    plt.show()
    doubleimgplot(figura, imgidentidade)
elif (escolha=='f'):
  imagemTopS= convolucao(np.array(figura), topsobel)
  imagemTopS
  print("Matriz de top sobel")
  print(topsobel)
  arraytops = np.array(imagemTopS, dtype=np.uint8)
  topsval=sns.heatmap(arraytops, cmap='gray',annot=True, fmt="d",cbar=False,yticklabels=False,xticklabels=False)
  plt.show()
  doubleimgplot(figura,imagemTopS)
elif (escolha=='g'):
    imgoutline = convolucao(np.array(figura), outline)
    imgoutline
    print("Matriz de outline")
    print(outline)
    arrayout = np.array(imgoutline, dtype=np.uint8)
    outval = sns.heatmap(arrayout, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False, xticklabels=False)
    plt.show()
    doubleimgplot(figura, imgoutline)
elif (escolha=='h'):
    imagemLefS = convolucao(np.array(figura), leftsobel)
    imagemLefS
    print("matriz de left sobel")
    print(leftsobel)
    arraylefs = np.array(imagemLefS, dtype=np.uint8)
    lefsval = sns.heatmap(arraylefs, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False, xticklabels=False)
    plt.show()
    doubleimgplot(figura, imagemLefS)
elif (escolha=='i'):
    imagemRigS = convolucao(np.array(figura), rightsobel)
    imagemRigS
    print("Matriz de right sobel")
    print(rightsobel)
    arrayrigs = np.array(imagemRigS, dtype=np.uint8)
    rigsval = sns.heatmap(arrayrigs, cmap='gray', annot=True, fmt="d", cbar=False, yticklabels=False, xticklabels=False)
    plt.show()
    doubleimgplot(figura, imagemRigS)
elif (escolha=='j'):
  suaarray = []
  for i in range(9):
    suaarray.append(float(input("Elemento:")))
  suaarray = numpy.array(suaarray)
  imgresultante= convolucao(np.array(figura), suaarray)
  imgresultante
  print("Matriz escolhida:")
  print(suaarray)
  arraycra = np.array(imgresultante, dtype=np.uint8)
  createval=sns.heatmap(arraycra, cmap='gray',annot=True, fmt="d",cbar=False,yticklabels=False,xticklabels=False)
  plt.show()
  doubleimgplot(figura,imgresultante)
