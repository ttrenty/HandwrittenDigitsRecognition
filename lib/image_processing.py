#-*- coding: utf-8 -*-
"""
Created on 2021-04-03
@author: ttrenty
"""

import  matplotlib.pyplot as plt
import numpy as np
import time
from os import path
from PIL import Image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~IMAGE PROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def readImage(path):
    image = plt.imread(path)
    return image

def PILtoNP(image):
    image = np.array(image)
    return(image)

def greyScale(image):
    row = image.shape[0]
    column = image.shape[1]
    newImage = np.zeros((row,column))

    for i  in range(row):
        for ii  in range(column):
            r,g,b = image[i,ii, 0],image[i,ii, 1],image[i,ii, 2]
            newImage[i,ii] = (0.4*r+0.3*g+0.4*b)/3

    return newImage

def getAverageGrey(image):
    row =image.shape[0]
    column = image.shape[1]
    totalPixel = row  * column

    greys = 0
    for i  in range(row):
        for ii  in range(column):
            greys += image[i,ii]

    return greys / totalPixel

def blackAndWhite(image, debug): #image has to be greyscale
    row =image.shape[0]
    column = image.shape[1]
    #totalPixel = row  * column

    #determine average grey level
    averageGrey = getAverageGrey(image)
    if debug: print('average grey :', averageGrey)

    newImage = np.zeros((row,column))

    for i  in range(row):
        for ii  in range(column):
            #black --> 0 ; white --> 1
            if image[i,ii] < averageGrey:
                newImage[i,ii] = 0
            else:
                newImage[i,ii] = 1

    return newImage


def cutImageToMatchNumber(image): #imageBW

    row = image.shape[0]
    column = image.shape[1]
    #if debug==True: print("row size :", row, "; column size :", column )

    #we search the first with Est and South, to determine top left and top right
    #similarly we search the last with Est and North, to determine bottom left and bottom right

    r = 0
    p = 0
    while np.sum(image[r])/column > 0.9999:
        r += 1
        p += 1

    topLeft = False

    while topLeft == False:

        c=0
        #print(['WEST','EST', 'NORTH', 'SOUTH'])
        while topLeft == False and c < column:
            pixel = image[r, c]
            if pixel == 0:
                if topLeft == False:
                    topLeftCoordinates = [r,c]
                    topLeft = True

            if pixel == 1:
                Around = getAround(image, r, c)

                if Around[1] == True and Around[3] == True and topLeft == False:
                    topLeftCoordinates = [r,c]
                    topLeft = True

            c+=1
        c=0
        r+=1

    r = row-1
    p = 0
    while np.sum(image[r])/column > 0.9999:
        r-=1
        p-=1

    bottomRight = False

    while bottomRight == False:

        c=column-1

        while bottomRight == False and c >= 0:
            pixel = image[r, c]

            if pixel == 0:
                bottomRightCoordinates = [r,c]
                bottomRight = True

            if pixel == 1:
                Around = getAround(image, r, c)

                if Around[0] == True and Around[2] == True:
                    bottomRight = True
                    bottomRightCoordinates = [r,c]
            c-=1
        c=column-1
        r-=1

    return topLeftCoordinates, bottomRightCoordinates

def colorImage(image, BWimage): #image and BW image of same size

    row =image.shape[0]
    column = image.shape[1]
    dimension = len(image[0,0])
    for r  in range(row):
        for c  in range(column):

            pixel = BWimage[r,c]
            if pixel == 0:
                if dimension==4:
                    image[r,c] = [0,0,0,255]
                else:
                    image[r,c] = [0,0,0]

            if pixel == 1:

                #determine correct cavity type

                Around = getAround(BWimage, r, c)


                #give proper property to the pixel

                """
                Exemple de la cavité OUEST :un pixel blanc appartient à une cavité OUEST
                lorsqu’il est encadrés par des pixels noirs au NORD au SUD et à l’EST.
                """

                """
                Exemple de la cavité NORD :un pixel blanc appartient à une cavité NORD
                lorsqu’il est encadré par des pixels noirs à l’EST, à l'OUEST et au SUD.
                """

                #Features --> [nO, nE (nWest), nN, nS, nC]
                #Around --> ['WEST', 'EST', 'NORTH', 'SOUTH']

                if Around[0] == True and Around[1] == True and Around[2] == True and Around[3] == True:
                    #CENTER
                    if dimension==4:
                        image[r,c] = [255,0,255,255]
                    else:
                        image[r,c] = [255,0,255]


                elif Around[1] == True and Around[2] == True and Around[3] == True:
                    #WEST
                    if dimension==4:
                        image[r,c] = [0,255,0,255]
                    else:
                        image[r,c] = [0,255,0]



                elif Around[0] == True and Around[2] == True and Around[3] == True:
                    #EST
                    if dimension==4:
                        image[r,c] = [255,0,0,255]
                    else:
                        image[r,c] = [255,0,0]



                elif  Around[0] == True and Around[1] == True and Around[3] == True:
                    #NORTH
                    if dimension==4:
                        image[r,c] = [0,0,255,255]
                    else:
                        image[r,c] = [0,0,255]



                elif Around[0] == True and Around[1] == True and Around[2] == True :
                    #SOUTH
                    if dimension==4:
                        image[r,c] = [255*0.7,255*0.7,255,255]
                    else:
                        image[r,c] = [255*0.7,255*0.7,255]

                else:
                    if dimension==4:
                        image[r,c] = [255,255,255,255]
                    else:
                        image[r,c] = [255,255,255]

    return(image)

def colorImageAround1(image, BWimage): #image and BW image of same size

    newImage = np.copy(image)

    row =image.shape[0]
    column = image.shape[1]
    dimension = len(newImage[0,0])

    for r  in range(row):
        for c  in range(column):

            pixel = BWimage[r,c]
            if pixel == 0:
                if dimension==4:
                    newImage[r,c] = [0,0,0,255]
                else:
                    newImage[r,c] = [0,0,0]

            if pixel == 1:
                #determine correct cavity type

                Around = getAround(BWimage, r, c)

                #give proper property to the pixel

                """
                Exemple de la cavité OUEST :un pixel blanc appartient à une cavité OUEST
                lorsqu’il est encadrés par des pixels noirs au NORD au SUD et à l’EST.
                """

                """
                Exemple de la cavité NORD :un pixel blanc appartient à une cavité NORD
                lorsqu’il est encadré par des pixels noirs à l’EST, à l'OUEST et au SUD.
                """

                #Features --> [nO, nE (nWest), nN, nS, nC]
                #Around --> ['WEST', 'EST', 'NORTH', 'SOUTH']

                if Around[2] == True and Around[3] == True:
                    #NORTH
                    if dimension==4:
                        newImage[r,c] = [255,0,255,255]
                    else:
                        newImage[r,c] = [255,0,255]

                elif Around[2] == True :
                    #NORTH
                    if dimension==4:
                        newImage[r,c] = [255,0,0,255]
                    else:
                        newImage[r,c] = [255,0,0]


                elif  Around[3] == True :
                    #SOUTH
                    if dimension==4:
                        newImage[r,c] = [0,0,255,255]
                    else:
                        newImage[r,c] = [0,0,255]

                else:
                    if dimension==4:
                        newImage[r,c] = [255,255,255,255]
                    else:
                        newImage[r,c] = [255,255,255]


    return(newImage)

def colorImageAround2(image, BWimage): #image and BW image of same size

    newImage = np.copy(image)
    row =newImage.shape[0]
    column = newImage.shape[1]
    dimension = len(newImage[0,0])
    for r  in range(row):
        for c  in range(column):

            pixel = BWimage[r,c]
            if pixel == 0:
                if dimension==4:
                    newImage[r,c] = [0,0,0,255]
                else:
                    newImage[r,c] = [0,0,0]

            if pixel == 1:
                #determine correct cavity type

                Around = getAround(BWimage, r, c)

                #give proper property to the pixel

                """
                Exemple de la cavité OUEST :un pixel blanc appartient à une cavité OUEST
                lorsqu’il est encadrés par des pixels noirs au NORD au SUD et à l’EST.
                """

                """
                Exemple de la cavité NORD :un pixel blanc appartient à une cavité NORD
                lorsqu’il est encadré par des pixels noirs à l’EST, à l'OUEST et au SUD.
                """

                #Features --> [nO, nE (nWest), nN, nS, nC]
                #Around --> ['WEST', 'EST', 'NORTH', 'SOUTH']

                if Around[0] == True and Around[1] == True:
                    #NORTH
                    if dimension==4:
                        newImage[r,c] = [255,0,255,255]
                    else:
                        newImage[r,c] = [255,0,255]

                elif Around[0] == True :
                    #NORTH
                    if dimension==4:
                        newImage[r,c] = [255,0,0,255]
                    else:
                        newImage[r,c] = [255,0,0]


                elif  Around[1] == True :
                    #SOUTH
                    if dimension==4:
                        newImage[r,c] = [0,0,255,255]
                    else:
                        newImage[r,c] = [0,0,255]



                else:
                    if dimension==4:
                        newImage[r,c] = [255,255,255,255]
                    else:
                        newImage[r,c] = [255,255,255]


    return(newImage)

def showImage(image, title = None):
    fig, ax = plt.subplots(dpi=250)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    if title:
        plt.title(title)
    plt.show()

def divideImageBy2(image, debug):
    row = image.shape[0]
    column = image.shape[1]

    if len(image.shape)>2:
        rgb = image.shape[2]
        newImage = np.empty((0,column, rgb), int)

    else:
        newImage = np.empty((0,column), int)


    for r in range(row):
        if r%2 == 0:
            newImage = np.append(newImage, [image[r]] , axis=0)

    deletedC = 0
    for c in range(column):
        if c%2 != 0:
            newImage = np.delete(newImage, c-deletedC, 1)
            deletedC += 1


    if debug: print('row and column reduced by 2')

    return newImage

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FEATURES EXTRATOR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#complexity to improve!!!!!

def getAround(image, r, c):
    Around = np.array([False for i in range(4)]) #bool for black pixel either on West, Est, North or South

    row =image.shape[0]
    column = image.shape[1]

    #check West
    checkC = c-1
    while checkC >= 0 and Around[0] == False:
        if image[r, checkC] == 0:
            #print('thing on right of ', r,c)
            Around[0] = True
        checkC -= 1

    #check Est
    checkC = c+1
    while checkC < column and Around[1] == False:

        if image[r, checkC] == 0:
            Around[1] = True
        checkC += 1

    #check North
    checkR = r-1

    while checkR >= 0 and Around[2] == False:
        #print(r,c,'is north')
        if image[checkR, c] == 0:
            Around[2] = True
        checkR -= 1

    #check South
    checkR = r+1
    while checkR < row and Around[3] == False:
        if image[checkR, c] == 0:
            Around[3] = True
        checkR += 1

    return(Around)


def getFeatures(image, debug):
    row =image.shape[0]
    column = image.shape[1]

    Features = [0]*5 # represent nO, nE (nWest), nN, nS, nC
    #print(Features)
    compt = 0

    for r  in range(row):
        for c  in range(column):
            pixel = image[r, c]
            #if pixel == 0:
                #print('black',r,c)

            if pixel == 1:
                compt += 1

                #determine correct cavity type

                Around = getAround(image, r, c)
                # print(Around)

                #give proper property to the pixel

                """
                Exemple de la cavité OUEST :un pixel blanc appartient à une cavité OUEST
                lorsqu’il est encadrés par des pixels noirs au NORD au SUD et à l’EST.
                """

                """
                Exemple de la cavité NORD :un pixel blanc appartient à une cavité NORD
                lorsqu’il est encadré par des pixels noirs à l’EST, à l'OUEST et au SUD.
                """

                #Features --> [nO, nE (nWest), nN, nS, nC]
                #Around --> ['WEST', 'EST', 'NORTH', 'SOUTH']

                if Around[0] == True and Around[1] == True and Around[2] == True and Around[3] == True:
                    #CENTER
                    #print(r,c, 'is CENTER')
                    Features[4] +=1

                elif Around[1] == True and Around[2] == True and Around[3] == True:
                    #WEST
                    #print(r,c, 'is WEST')
                    Features[0] +=1

                elif Around[0] == True and Around[2] == True and Around[3] == True:
                    #EST
                    #print(r,c, 'is EST')
                    Features[1] +=1

                elif  Around[0] == True and Around[1] == True and Around[3] == True:
                    #NORTH
                    #print(r,c, 'is NORTH')
                    Features[2] +=1

                elif Around[0] == True and Around[1] == True and Around[2] == True :
                    #SOUTH
                    #print(r,c, 'is SOUTH')
                    Features[3] +=1



    if debug: print(["WEST","EST","NORTH","SOUTH","CENTER"])
    if debug: print(Features[0],Features[1],Features[2],Features[3],Features[4])

    #print(compt)
    #print(Features)
    for i in range(len(Features)):
        Features[i] /= (row*column)

    if debug: print(["nW","nE","nN","nS","nC"])
    if debug: print(Features)

    return Features



# def showAroundinTerminal(image, debug):
#     row =image.shape[0]
#     column = image.shape[1]
#     for r in range(row):
#         if debug: print(['WEST','EST', 'NORTH', 'SOUTH'])
#         for c in range(column):
#             pixel = image[r, c]

#             if pixel == 1:
#                 if debug:print(r,c)
#                 Around = getAround(image, r, c)
#                 if debug: print(Around)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MAIN FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def showSteps(filename, debug):
    path = './data/testing/'+filename+'.png'

    PILimage = Image.open(path)

    image = PILtoNP(PILimage)
    showImage(image, 'base png')

    PILimage = change_contrast(PILimage, 200)
    image = PILtoNP(PILimage)
    showImage(image, 'png higher contrast')

    # print('size', image.shape[0] * image.shape[0], image.shape[0] * image.shape[0] > 1000)
    if debug: print('shape of original png :', (image.shape[0], image.shape[1]))
    if debug: print('total pixels of original png :', image.shape[0] * image.shape[1])
    while image.shape[0] * image.shape[0] > 30000:
        image =  divideImageBy2(image, debug)
        showImage(image, 'resized higher contrast')


    greyImage = greyScale(image)
    # showImage(greyImage, 'grey scale')

    BWimage = blackAndWhite(greyImage, debug)
    showImage(BWimage, 'Black and White')


    topLeftCoordinates, bottomRightCoordinates = cutImageToMatchNumber(BWimage)
    BWimage = BWimage[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
    showImage(BWimage, 'cropped Black and White')

    image = image[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
    coloredImageAround = colorImageAround1(image, BWimage)
    showImage(coloredImageAround, 'show Around')

    coloredImage = colorImage(image, BWimage)
    showImage(coloredImage, 'show features')

    Features = getFeatures(BWimage, debug)

    return Features



def resizeImageTesting(filename, debug):
    path = './data/testing/'+filename+'.png'

    PILimage = Image.open(path)

    image = PILtoNP(PILimage)
    showImage(image, 'base png')

    PILimage = change_contrast(PILimage, 200)
    image = PILtoNP(PILimage)
    showImage(image, 'png higher contrast')


    if debug: print('shape of original png :', (image.shape[0], image.shape[1]))
    if debug: print('total pixels of original png :', image.shape[0] * image.shape[1])

    while image.shape[0] * image.shape[0] > 100000:
        image =  divideImageBy2(image, debug)
        showImage(image, 'resized higher contrast')


    greyImage = greyScale(image)
    # showImage(greyImage, 'grey scale')

    BWimage = blackAndWhite(greyImage, debug)
    showImage(BWimage, 'Black and White')


    topLeftCoordinates, bottomRightCoordinates = cutImageToMatchNumber(BWimage)
    BWimage = BWimage[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
    showImage(BWimage, 'cropped Black and White')


    image = image[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]

    if debug:
        coloredImageAround = colorImageAround2(image, BWimage)
        showImage(coloredImageAround, 'Around Est and West --> rows')

    return(image, BWimage)

def resizeImageTraining(filename, debug):
    path = './data/training/user/'+filename+'.png'

    PILimage = Image.open(path)

    image = PILtoNP(PILimage)
    showImage(image, 'base png')

    PILimage = change_contrast(PILimage, 200)
    image = PILtoNP(PILimage)
    showImage(image, 'png higher contrast')


    if debug: print('shape of original png :', (image.shape[0], image.shape[1]))
    if debug: print('total pixels of original png :', image.shape[0] * image.shape[1])

    while image.shape[0] * image.shape[0] > 100000:
        image =  divideImageBy2(image, debug)
        showImage(image, 'resized higher contrast')


    greyImage = greyScale(image)
    # showImage(greyImage, 'grey scale')

    BWimage = blackAndWhite(greyImage, debug)
    showImage(BWimage, 'Black and White')


    topLeftCoordinates, bottomRightCoordinates = cutImageToMatchNumber(BWimage)
    BWimage = BWimage[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
    showImage(BWimage, 'cropped Black and White')


    image = image[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]

    if debug:
        coloredImageAround = colorImageAround2(image, BWimage)
        showImage(coloredImageAround, 'Around Est and West --> rows')

    return(image, BWimage)


def separateNumbers(image, BWimage, debug):

    LinesImages, LinesImagesBW =  getLines(image, BWimage, debug)
    if debug: print('total number of lines :', len(LinesImages))

    DigitsImages = []
    DigitsImagesBW = []
    LineImagesWithBorder = np.array([])
    for i in range(len(LinesImages)):
        if debug:
            coloredImageAround = colorImageAround1(LinesImages[i], LinesImagesBW[i])
            showImage(coloredImageAround, 'Around North and South --> digits')

        digits = getDigits(LinesImages[i], LinesImagesBW[i], i)
        DigitsImages += digits[0]
        DigitsImagesBW  += digits[1]

        imageWithBorder = digits[2]

        #LineImagesWithBorder
        if LineImagesWithBorder.size == 0:
            LineImagesWithBorder = np.copy(imageWithBorder)
        else:
            biggest = max(LineImagesWithBorder.shape[1], imageWithBorder.shape[1])

            if LineImagesWithBorder.shape[1] == biggest:
                for u in range(biggest-imageWithBorder.shape[1] ):
                    newColumn=np.ones((imageWithBorder.shape[0], 1, 3))*255
                    imageWithBorder = np.column_stack((imageWithBorder, newColumn))

            else:
                for u in range(biggest-LineImagesWithBorder.shape[1] ):
                    newColumn=np.ones((LineImagesWithBorder.shape[0], 1, 3))*255
                    LineImagesWithBorder = np.column_stack((LineImagesWithBorder, newColumn))

            LineImagesWithBorder = np.concatenate((LineImagesWithBorder, imageWithBorder))


        DigitsImages += ['newLine']
        DigitsImagesBW += ['newLine']

    showImage(LineImagesWithBorder,'digits detected.')

    if debug:  print('total number of digits :', len(DigitsImages))

    return DigitsImages, DigitsImagesBW

def getLines(image, imageBW, debug):

    row = image.shape[0]
    column = image.shape[1]


    #get the top left and bottom right of each digits:

    #we detect a new number by seeing a white pixel has no pixel under it neither above it
    #since we already cropped the image, we only need to check the first line adn last line

    Coordonates = [[],[]] #[[topLeft(1), topLeft(2)..., topLeft(n)],[bottomRight(n), bottomRight(n-1)..., bottomRight(1)]]

    found = False
    gap = True

    for r in range(row):

        pixel = imageBW[r, 0]

        if pixel == 0:
            if found == False:
                Coordonates[0] += [[r,0]]
                found = True

        if pixel == 1:
            Around = getAround(imageBW, r, 0)
            if (Around[0] == True or Around[1] == True) and found == False and gap == True:
                Coordonates[0] += [[r,0]]
                found = True
                gap = False

            if gap == False and Around[0] == False and Around[1] == False:
                gap = True
                found = False

    found = False
    gap = True
    for r in range(row-1,-1,-1):

        pixel = imageBW[r, column-1]

        if pixel == 0:
            if found == False:
                Coordonates[1] += [[r,column-1]]
                found = True

        if pixel == 1:
            Around = getAround(imageBW, r, column-1)
            if (Around[0] == True or Around[1] == True) and found == False and gap == True:
                Coordonates[1] += [[r,column-1]]
                found = True
                gap = False

            if gap == False and Around[0] == False and Around[1] == False:
                gap = True
                found = False

    if debug: print("Coordonates", Coordonates)


    #return LinesImages, LinesImagesBW
    #crop the big numpy matrix for each digits:
    LinesImages = []
    LinesImagesBW = []


    for i in range(len(Coordonates[0])):
        topLeftCoordinates = Coordonates[0][i]
        bottomRightCoordinates = Coordonates[1][-(i+1)]
        #if debug==True: print("new cut", topLeftCoordinates, bottomRightCoordinates)
        linesImages = image[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
        linesImagesBW = imageBW[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]


        digitTopLeftCoordinates, digitBottomRightCoordinates = cutImageToMatchNumber(linesImagesBW)

        linesImages = linesImages[digitTopLeftCoordinates[0]:digitBottomRightCoordinates[0]+1,digitTopLeftCoordinates[1]:digitBottomRightCoordinates[1]+1]
        linesImagesBW = linesImagesBW[digitTopLeftCoordinates[0]:digitBottomRightCoordinates[0]+1,digitTopLeftCoordinates[1]:digitBottomRightCoordinates[1]+1]

        LinesImages += [linesImages]
        LinesImagesBW += [linesImagesBW]

    # verify
    imageWithBorder = np.copy(image)
    dimension = len(imageWithBorder[0,0])


    for u in range(len(Coordonates[0])):
        point  = Coordonates[0][u]

        if dimension == 3:

            for v in range(Coordonates[1][-(u+1)][0]-point[0]):
                if imageWithBorder[point[0]+v, point[1]][0] == 255:
                    imageWithBorder[point[0]+v, point[1]] = [255,0,0]
                if imageWithBorder[point[0]+v, Coordonates[1][-(u+1)][1]][0] == 255:
                    imageWithBorder[point[0]+v, Coordonates[1][-(u+1)][1]] = [255,0,0] #just to see the border

            for v in range(Coordonates[1][-(u+1)][1]+1):
                if imageWithBorder[point[0], v][0] == 255:
                    imageWithBorder[point[0], v] = [255,0,0]
                if imageWithBorder[Coordonates[1][-(u+1)][0], v][0] == 255:
                    imageWithBorder[Coordonates[1][-(u+1)][0], v] = [255,0,0] #just to see the border


        else:
            imageWithBorder[list(point)[0], list(point)[1]] = [255,0,0,255] #just to see the border

    showImage(imageWithBorder,str(len(Coordonates[0]))+' lines detected')

    return [LinesImages, LinesImagesBW]

def getDigits(lineImage, lineImageBW, n):
    row = lineImage.shape[0]
    column = lineImage.shape[1]

    #get the top left and bottom right of each digits:

    #we detect a new number by seeing a white pixel has no pixel under it neither above it
    #since we already cropped the image, we only need to check the first line adn last line

    Coordonates = [[],[]] #[[topLeft(1), topLeft(2)..., bottomRight(n)],[bottomRight(n), bottomRight(n-1)..., bottomRight(1)]]

    found = False
    gap = True

    for c  in range(column):

        pixel = lineImageBW[0, c]

        if pixel == 0:
            if found == False:
                Coordonates[0] += [[0,c]]
                found = True

        if pixel == 1:
            Around = getAround(lineImageBW, 0, c)
            if (Around[2] == True or Around[3] == True) and found == False and gap == True:
                Coordonates[0] += [[0,c]]
                found = True
                gap = False

            if gap == False and Around[2] == False and Around[3] == False:
                gap = True
                found = False

    found = False
    gap = True
    for c in range(column-1,-1,-1):

        pixel = lineImageBW[row-1, c]

        if pixel == 0:
            if found == False:
                Coordonates[1] += [[row-1,c]]
                found = True

        if pixel == 1:
            Around = getAround(lineImageBW, row-1, c)
            if (Around[2] == True or Around[3] == True) and found == False and gap == True:
                Coordonates[1] += [[row-1,c]]
                found = True
                gap = False

            if gap == False and Around[2] == False and Around[3] == False:
                gap = True
                found = False

    #crop the big numpy matrix for each digits:
    DigitsImages = []
    DigitsImagesBW = []

    for i in range(len(Coordonates[0])):
        topLeftCoordinates = Coordonates[0][i]
        bottomRightCoordinates = Coordonates[1][-(i+1)]
        #if debug==True:  print("new cut", topLeftCoordinates, bottomRightCoordinates)
        digitsImages = lineImage[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]
        digitsImagesBW = lineImageBW[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]

        digitTopLeftCoordinates, digitBottomRightCoordinates = cutImageToMatchNumber(digitsImagesBW)

        Coordonates[0][i][0] += digitTopLeftCoordinates[0]
        Coordonates[1][-(i+1)][0] = Coordonates[0][i][0] + (digitBottomRightCoordinates[0]-digitTopLeftCoordinates[0])

        digitsImages = digitsImages[digitTopLeftCoordinates[0]:digitBottomRightCoordinates[0]+1,digitTopLeftCoordinates[1]:digitBottomRightCoordinates[1]+1]
        digitsImagesBW = digitsImagesBW[digitTopLeftCoordinates[0]:digitBottomRightCoordinates[0]+1,digitTopLeftCoordinates[1]:digitBottomRightCoordinates[1]+1]

        DigitsImages += [digitsImages]
        DigitsImagesBW += [digitsImagesBW]

     # verify

    imageWithBorder = np.copy(lineImage)
    dimension = len(imageWithBorder[0,0])

    for u in range(len(Coordonates[0])):
        point  = Coordonates[0][u]


        if dimension == 3:
            #if debug==True: print(Coordonates[1][-(u+1)][0])
            for v in range(Coordonates[1][-(u+1)][0]-point[0]):
                if imageWithBorder[point[0]+v, point[1]][0] == 255:
                    imageWithBorder[point[0]+v, point[1]] = [255,0,0]
                if imageWithBorder[point[0]+v, Coordonates[1][-(u+1)][1]][0] == 255:
                    imageWithBorder[point[0]+v, Coordonates[1][-(u+1)][1]] = [255,0,0] #just to see the border

            #for red rows
            for v in range(Coordonates[1][-(u+1)][1]+1- point[1]):
                if imageWithBorder[point[0], point[1]+v][0] == 255:
                    imageWithBorder[point[0], point[1]+v] = [255,0,0]
                if imageWithBorder[Coordonates[1][-(u+1)][0], point[1]+v][0] == 255:
                    imageWithBorder[Coordonates[1][-(u+1)][0], point[1]+v] = [255,0,0] #just to see the border


    return DigitsImages, DigitsImagesBW, imageWithBorder



def showStepsforMatrix(newImage, imageMatrix, imageMatrixBW, debug):


    image = imageMatrix
    BWimage = imageMatrixBW


    coloredImage = colorImage(image, BWimage)

    if newImage.size == 0:
        newImage = np.copy(coloredImage)
    else:
        biggest = max(newImage.shape[0], coloredImage.shape[0])

        if newImage.shape[0] == biggest:
            for u in range(biggest-coloredImage.shape[0] ):
                newrow=np.ones((1, coloredImage.shape[1], 3))*255
                coloredImage = np.vstack([coloredImage, newrow])

        else:
            for u in range(biggest-newImage.shape[0] ):
                newrow=np.ones((1, newImage.shape[1], 3))*255
                newImage = np.vstack([newImage, newrow])


        newImage= np.column_stack((newImage, coloredImage))

    #showImage(newImage, 'newImage')
    # showImage(coloredImage, 'show features')

    # showAround(image)

    Features = getFeatures(BWimage, debug)

    return  newImage, Features


def vect(filename, debug):
    t1 = time.time()
    path = './data/training/digits/'+filename+'.png'

    image = readImage(path)
    greyImage = greyScale(image)
    BWimage = blackAndWhite(greyImage, debug)

    topLeftCoordinates, bottomRightCoordinates = cutImageToMatchNumber(BWimage)
    BWimage = BWimage[topLeftCoordinates[0]:bottomRightCoordinates[0]+1,topLeftCoordinates[1]:bottomRightCoordinates[1]+1]

    Features = getFeatures(BWimage, debug)
    if debug: print('elapsed time', time.time()-t1)
    return Features