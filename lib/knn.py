#-*- coding: utf-8 -*-
"""
Created on 2021-04-03
@author: ttrenty
"""

import math
import  matplotlib.pyplot as plt
import numpy as np
import json
from os import path
import seaborn as sns
import sys
sys.path.append('./')
from image_processing import vect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DATA AND GUESSES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def createDataSet(debug):
    # fileName = "TRAIN_DATA/dataSetAverage.json"
    # with open(fileName,'r') as json_file:
    #    Features = json.load(json_file)

    FeaturesAverages = {"0": {"total" : 0, "features" : [0,0,0,0,0]}, "1": {"total" : 0, "features" : [0,0,0,0,0]}, "2": {"total" : 0, "features" : [0,0,0,0,0]}, "3": {"total" : 0, "features" : [0,0,0,0,0]}, "4": {"total" : 0, "features" : [0,0,0,0,0]}, "5": {"total" : 0, "features" : [0,0,0,0,0]}, "6": {"total" : 0, "features" : [0,0,0,0,0]}, "7": {"total" : 0, "features" : [0,0,0,0,0]}, "8": {"total" : 0, "features" : [0,0,0,0,0]}, "9": {"total" : 0, "features" : [0,0,0,0,0]}}
    FeaturesKNN = {"0": [], "1": [],"2": [],"3": [],"4": [],"5": [],"6": [],"7": [],"8": [],"9": []}


    for number in range(10):
        p = 0
        filename = 'modele' + str(number) + '_' + str(p)
        pathe =  './data/training/digits/' + filename + '.png'

        while path.exists(pathe):
            if debug: print('------------------'+filename+'--------------------')
            newFeature = vect(filename, debug)
            #print(newFeature)
            #print(Features[str(number)]['features'])
            FeaturesAverages[str(number)]['features'] = (np.add(np.array(FeaturesAverages[str(number)]['features'])*FeaturesAverages[str(number)]['total'], newFeature) / (FeaturesAverages[str(number)]['total'] +1)).tolist()
            FeaturesAverages[str(number)]['total'] += 1

            FeaturesKNN[str(number)] += [newFeature]

            p+=1
            filename = 'modele' + str(number) + '_' + str(p)
            pathe =  './data/training/digits/' + filename + '.png'


    fileName = "./data/training/dataSetAverage.json"
    with open(fileName, 'w') as outfile:
        json.dump(FeaturesAverages, outfile)

    fileName = "./data/training/dataSetKNN.json"
    with open(fileName, 'w') as outfile:
        json.dump(FeaturesKNN, outfile)


def addToDataSetAverage(point, value):
    fileName = "./data/training/dataSetAverage.json"
    with open(fileName,'r') as json_file:
        Features = json.load(json_file)

    ######### check if point is not in knnDataSet #########

    Features[str(value)]['features'] = (np.add(np.array(Features[str(value)]['features'])*Features[str(value)]['total'], point) / (Features[str(value)]['total'] +1)).tolist()
    Features[str(value)]['total'] += 1
    print("point", value,"added in average dataSet, new total :", Features[str(value)]['total'])

    with open(fileName, 'w') as outfile:
        json.dump(Features, outfile)

def addToDataSetKNN(point, value):
    fileName = "./data/training/dataSetKNN.json"
    with open(fileName,'r') as json_file:
        Features = json.load(json_file)

    # if point not in Features[str(value)]:

    #     Features[str(value)] += [point]

    Features[str(value)] += [point]

    print("point", value, "added in KNN dataSet, new total :", len(Features[str(value)]))

    with open(fileName, 'w') as outfile:
        json.dump(Features, outfile)

def getEuclidienneDistanceR5(point1, point2):
    S =0

    for i in range(5):
        S += (point1[i] - point2[i]) ** 2

    return (math.sqrt(S))


def showHeatmapDataSet():
    fileName = "./data/training/dataSetAverage.json"
    with open(fileName,'r') as json_file:
       Features = json.load(json_file)

    D = np.zeros((10,10))
    #print(D)

    for i in range(len(Features)):
        feature = Features[str(i)]['features']
        for u in range(len(Features)):
            otherFeature = Features[str(u)]['features']
            d = getEuclidienneDistanceR5(feature, otherFeature)
            pourcent = (1-d)*100

            D[i, u] = pourcent

    # return D
    Max = np.amax(D)
    corr = D
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    mask = mask - np.identity(10)

    plt.figure(dpi=300) # width and height in inches
    sns.heatmap(corr, mask=mask, linewidth=0.5, vmax=Max, square=True,  cmap="Greens", annot=True, fmt='.0f',annot_kws={"size": 8})
    #plt.show()



def guessAverage(point):
    fileName = "./data/training/dataSetAverage.json"
    with open(fileName,'r') as json_file:
       Features = json.load(json_file)

    feature = Features['0']['features']
    Min = getEuclidienneDistanceR5(point, feature)

    number = '0'

    for i in range(1,len(Features)):
        feature = Features[str(i)]['features']
        if getEuclidienneDistanceR5(point, feature) < Min:
            Min = getEuclidienneDistanceR5(point, feature)
            number = i

    return [number, (1-Min)*100]


def guessKNN(point, k):
    fileName = "./data/training/dataSetKNN.json"
    with open(fileName,'r') as json_file:
       Features = json.load(json_file)

    feature = Features['0'][0]
    maxOfmin = getEuclidienneDistanceR5(point, feature)
    Min = [maxOfmin]*k
    #print(Min)
    maxOfmin = max(Min)
    Guesses = [0]*k
    #print(Min, Guesses, maxOfmin)

    for i in range(10):
        #print(Min, Guesses, maxOfmin)
        FeaturesNumber = Features[str(i)]
        for feature in FeaturesNumber:
            maxOfmin = max(Min)
            max_index = Min.index(maxOfmin)
            # print("in",point,feature)
            if getEuclidienneDistanceR5(point, feature) < maxOfmin:
                Min[max_index] = getEuclidienneDistanceR5(point, feature)
                Guesses[max_index] = i

    S = []
    for i in range(10):
        S += [Guesses.count(i)]

    #print(S)
    guess = max(S)
    #guessIndex = S.index(guess)
    guessIndex = [i for i, x in enumerate(S) if x == guess]

    #print(Guesses, guessIndex)

    return [guessIndex, guess]


def getResults(Features, k):
    guessedKNN = guessKNN(Features, k)
    guessedAverage = guessAverage(Features)



    #print(guessAverage)
    if len(guessedKNN[0]) == 1:
        print("\n KNN guess : It's a :", guessedKNN[0][0], ", accuracy :", guessedKNN[1],"/", k ," .")
        print("\n Average guess :  It's a :", guessedAverage[0], ", I'm", "{:.2f}".format(guessedAverage[1]),"% sure about that.")

        goodguess = input('Is my KNN guess correct? (y) / (n) : ')
        if goodguess == 'y':
            addToDataSetKNN(Features,guessedKNN[0][0])
            addToDataSetAverage(Features,guessedKNN[0][0])
        else:
            goodguess = input('Is my average guess correct tho? (y) / (n) : ')
            if goodguess == 'y':
                addToDataSetKNN(Features,guessedAverage[0])
                addToDataSetAverage(Features,guessedAverage[0])
            else:
                rightGuess = input('Oh.. What was your number? ')
                addToDataSetKNN(Features,rightGuess)
                addToDataSetAverage(Features,rightGuess)

    else:
        print("\n KNN guess : I couldn't decide between :")
        for i in range(len(guessedKNN[0])-1):
            print(guessedKNN[0][i],'or,')
        print(guessedKNN[0][-1],'.')

        print("\n Average guess :  It's a :", guessedAverage[0], ", I'm", "{:.2f}".format(guessedAverage[1]),"% sure about that.")
        goodguess = input('Is my average guess correct tho? (y) / (n) : ')

        if goodguess=='y':
            addToDataSetKNN(Features,guessedAverage[0])
            addToDataSetAverage(Features,guessedAverage[0])
        else:
            rightGuess = input('What was your number actually? ')
            addToDataSetKNN(Features,rightGuess)
            addToDataSetAverage(Features,rightGuess)

def getResultsForMultipleDigits(Features, k):
    guessesKNN = []
    guessesAverage = []

    # print("Features", Features)

    uniqueGuessKNN = True

    for i in range(len(Features)):
        #print("here",Features[i])
        if Features[i] != []:

            guessesKNN += [guessKNN(Features[i], k)]
            if len(guessKNN(Features[i], k)[0]) != 1:
                uniqueGuessKNN = False
            guessesAverage += [guessAverage(Features[i])]
        else:
            guessesKNN += [' ']
            guessesAverage += [' ']
    # print(guessesKNN, guessesAverage)

    if uniqueGuessKNN :
        guesKNN = ''
        accuracy = 0
        for i in range(len(guessesKNN)):
            if guessesKNN[i] != ' ':
                # print(guessesKNN[i][0][0],type(guessesKNN[i][0][0]))
                guesKNN += str(guessesKNN[i][0][0])
                accuracy += guessesKNN[i][1]
            else:
                guesKNN += ' '
        print("\n KNN guess : It's :", guesKNN, ", accuracy :", accuracy,"/", len(Features)*k ," .")

        guesAverage = ''
        percent = 0
        for i in range(len(guessesAverage)):
            if guessesAverage[i] != ' ':
                guesAverage += str(guessesAverage[i][0])
                percent += guessesAverage[i][1]
            guesAverage += ' '

        print("\n Average guess :  It's :", guesAverage, ", I'm", "{:.2f}".format(percent/len(Features)),"% sure about that.")


        goodguess = input('Is my KNN guess correct? (y) / (n) : ')
        if goodguess == 'y':
            for i in range(len(guessesKNN)):
                if guessesKNN[i] != ' ':
                    # print(Features[i], guessesKNN[i][0][0])
                    addToDataSetKNN(Features[i],guessesKNN[i][0][0])
                    addToDataSetAverage(Features[i],guessesKNN[i][0][0])
        else:
            goodguess = input('Is my average guess correct tho? (y) / (n) : ')
            if goodguess == 'y':
                for i in range(len(guessesAverage)):
                    if guessesAverage[i] != ' ':
                        addToDataSetKNN(Features[i],guessesAverage[i][0])
                        addToDataSetAverage(Features[i],guessesAverage[i][0])
            else:
                rightGuesses = input('Oh.. What was your whole number (with spaces in betwen the lines)? ')
                for i in range(len(rightGuesses)):
                    if rightGuesses[i] != ' ':
                        rightGuess = rightGuesses[i]
                        # print(Features[i],rightGuess)
                        addToDataSetKNN(Features[i],rightGuess)
                        addToDataSetAverage(Features[i],rightGuess)

    else:

        print("\n KNN guess : I couldn't decide between multiple numbers")
        #print("my knn guess", guessesKNN)

        # #for :
        # guessesKNN = [[[0,5], 5], [[1,5,6], 5], [[2], 4], [[1, 3], 2]]
        # #I want to get ['0121', '0123', '5121', '5123']

        guesKNN=['']
        # print(guessesKNN)

####NEED A REWORK
        for i in range(len(guessesKNN)):
            if guessesKNN[i] != ' ':
                if len(guessesKNN[i][0]) == 1:
                    for u in range(len(guesKNN)):
                        guesKNN[u] += str(guessesKNN[i][0][0])

                else:
                    guesKNN = [guesKNN[u] for u in range(len(guesKNN)) for o in range(len(guessesKNN[i][0]))]

                    p = 0
                    for u in range(len(guesKNN)):
                        guesKNN[u] += str(guessesKNN[i][0][p])
                        p += 1
                        if p >= len(guessesKNN[i][0]):
                            p = 0
            else:
                for u in range(len(guesKNN)):
                    guesKNN[u] += ' '


        # print("heeeeere", guesKNN)

        for i in range(len(guesKNN)):
            print("guess",i+1,":", guesKNN[i])


        guesAverage = ''
        percent = 0
        for i in range(len(guessesAverage)):
            if guessesAverage[i] != ' ':
                guesAverage += str(guessesAverage[i][0])
                percent += guessesAverage[i][1]
            else:
                guesAverage += ' '

        print("\n Average guess :  It's :", guesAverage, ", I'm", "{:.2f}".format(percent/len(Features)),"% sure about that.")


        goodguess = input('Is one of my KNN guess correct at least? (y) / (n) : ')

        if goodguess == 'y':
            whichOne = int(input('Which one is it? (give its number) : '))
            #print("len",len(Features),len(guesKNN[whichOne-1]))
            for i in range(len(guesKNN[whichOne-1])):
                if guesKNN[whichOne-1][i]!= ' ':
                    addToDataSetKNN(Features[i],guesKNN[whichOne-1][i])
                    addToDataSetAverage(Features[i],guesKNN[whichOne-1][i])


        else:
            goodguess = input('Is my average guess correct tho? (y) / (n) : ')
            if goodguess == 'y':
                for i in range(len(guessesAverage)):
                    if guessesAverage[i] != ' ':
                        addToDataSetKNN(Features[i],guessesAverage[i][0])
                        addToDataSetAverage(Features[i],guessesAverage[i][0])

            else:

                rightGuesses = input('Oh.. What was your whole number (with spaces in betwen the lines)? ')

                if len(rightGuesses) == len(guesAverage[:-1]):
                    for i in range(len(rightGuesses)):
                        if rightGuesses[i] != ' ':
                            rightGuess = rightGuesses[i]
                            # print(Features[i],rightGuess)
                            addToDataSetKNN(Features[i],rightGuess)
                            addToDataSetAverage(Features[i],rightGuess)

                while len(rightGuesses) != len(guesAverage[:-1]):
                    print("longueur donnée :", len(rightGuesses), "; longueur attendue :", len(guesAverage[:-1]))
                    rightGuesses = input('Oh your number is not in the correct format please try again (take care to the spaces and the number of digits pls) : ')
                    print(len(rightGuesses),len(guesAverage))
                    if len(rightGuesses) == len(guesAverage):
                        for i in range(len(rightGuesses)):
                            if rightGuesses[i] != ' ':
                                rightGuess = rightGuesses[i]
                                # print(Features[i],rightGuess)
                                addToDataSetKNN(Features[i],rightGuess)
                                addToDataSetAverage(Features[i],rightGuess)