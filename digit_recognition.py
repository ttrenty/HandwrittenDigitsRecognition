# -*- coding: utf-8 -*-
"""
Created on 2021-04-03
@author: ttrenty
"""

import numpy as np
from os import path
import sys
sys.path.append('./lib/')
from image_processing import *
from knn import *

debug = int(input("Do you want to track the program's progress in the console? (0 --> no, 1 --> yes): "))
if debug == 1:
    debug = True
else:
    debug = False

# ----by default, debug is off and no numbers are shown with pyplot----#

# debug = True
# showNumbers = True

# showHeatmapDataSet()

k = 3  # parameter for the KNN function
filename = '11111_22222_33333_44444_55555_66666_77777_88888_99999'

if not (path.exists("./data/training/dataSetAverage.json")):
    print("\n~~~~~~~~ Creating the dataDataset... ~~~~~~~~")
    createDataSet(debug)
    print("\n~~~~~~~~ Default dataset created successfully! ~~~~~~~~")
else:
    print("\n~~~~~~~~ KNN dataset found successfully! ~~~~~~~~")

ownImage = int(input("Do you want to add your own training image? (see existing example in ./data/training/user/) (0 --> no, 1 --> yes): "))

if ownImage:
    filename = input("What is the name of your image? (e.g., 'ttrenty' (without the .png)): ")
    filenamePath = "./data/training/user/" + filename + ".png"

    if not (path.exists(filenamePath)):
        print("Please check that your file is in the ./data/training/user/ folder (or rerun the script)")

    else:
        image, BWimage = resizeImageTraining(filename, debug)
        ImageMatrices, ImageMatricesBW = separateNumbers(image, BWimage, debug)

        if len(ImageMatrices) == 2:
            Features = showSteps(filename, debug)
            getResults(Features, k)

        else:  # -----------if multiple digits----------#

            Features = []
            imageTotalColored = np.array([])
            imageTotalColoredLine = np.array([])

            correctCutting = int(
                input("Is there exactly 1 digit per red box with only red boxes that frame the digits fairly closely? (0 --> no, 1 --> yes): "))
            if not (correctCutting):
                print("Sorry, you cannot use this script with this image")
            else:
                for i in range(len(ImageMatrices)):
                    if debug == True: print("\n~~~~~~~~ Digit number:", i + 1, "~~~~~~~~")
                    if not np.array_equal(ImageMatrices[i], 'newLine'):
                        imageTotalColoredLine, Feature = showStepsforMatrix(imageTotalColoredLine, ImageMatrices[i],
                                                                            ImageMatricesBW[i], debug)
                        Features += [Feature]
                    else:
                        if imageTotalColored.size == 0:
                            imageTotalColored = np.copy(imageTotalColoredLine)
                        else:

                            biggest = max(imageTotalColored.shape[1], imageTotalColoredLine.shape[1])

                            if imageTotalColored.shape[1] == biggest:
                                for u in range(biggest - imageTotalColoredLine.shape[1]):
                                    newColumn = np.ones((imageTotalColoredLine.shape[0], 1, 3)) * 255
                                    imageTotalColoredLine = np.column_stack((imageTotalColoredLine, newColumn))

                            else:
                                for u in range(biggest - imageTotalColored.shape[1]):
                                    newColumn = np.ones((imageTotalColored.shape[0], 1, 3)) * 255
                                    imageTotalColored = np.column_stack((imageTotalColored, newColumn))

                            imageTotalColored = np.concatenate((imageTotalColored, imageTotalColoredLine))

                        imageTotalColoredLine = np.array([])

                        Features += [[]]

                showImage(imageTotalColored)
                getResultsForMultipleDigits(Features, k)

else:
    print("\n~~~~~~~~ You have entered TEST mode ~~~~~~~~")
    filename = input("What is the name of the image to extract digits from? (e.g., 'full' (without the .png)): ")
    filenamePath = "./data/testing/" + filename + ".png"

    if not (path.exists(filenamePath)):
        print("Please check that your file is in the ./data/testing/ folder (or rerun the script)")

    else:
        image, BWimage = resizeImageTesting(filename, debug)
        ImageMatrices, ImageMatricesBW = separateNumbers(image, BWimage, debug)

        if len(ImageMatrices) == 2:
            Features = showSteps(filename, debug)
            getResults(Features, k)

        else:  # -----------if multiple digits----------#

            Features = []
            imageTotalColored = np.array([])
            imageTotalColoredLine = np.array([])

            correctCutting = int(
                input("Is there exactly 1 digit per red box with only red boxes that frame the digits fairly closely? (0 --> no, 1 --> yes): "))
            if not (correctCutting):
                print("Sorry, you cannot use this script with this image")
            else:
                for i in range(len(ImageMatrices)):
                    print("\n~~~~~~~~ Digit number:", i + 1, "~~~~~~~~")
                    if not np.array_equal(ImageMatrices[i], 'newLine'):
                        imageTotalColoredLine, Feature = showStepsforMatrix(imageTotalColoredLine, ImageMatrices[i],
                                                                            ImageMatricesBW[i], debug)
                        Features += [Feature]
                    else:
                        if imageTotalColored.size == 0:
                            imageTotalColored = np.copy(imageTotalColoredLine)
                        else:

                            biggest = max(imageTotalColored.shape[1], imageTotalColoredLine.shape[1])

                            if imageTotalColored.shape[1] == biggest:
                                for u in range(biggest - imageTotalColoredLine.shape[1]):
                                    newColumn = np.ones((imageTotalColoredLine.shape[0], 1, 3)) * 255
                                    imageTotalColoredLine = np.column_stack((imageTotalColoredLine, newColumn))

                            else:
                                for u in range(biggest - imageTotalColored.shape[1]):
                                    newColumn = np.ones((imageTotalColored.shape[0], 1, 3)) * 255
                                    imageTotalColored = np.column_stack((imageTotalColored, newColumn))

                            imageTotalColored = np.concatenate((imageTotalColored, imageTotalColoredLine))

                        imageTotalColoredLine = np.array([])

                        Features += [[]]

                showImage(imageTotalColored)
                getResultsForMultipleDigits(Features, k)
