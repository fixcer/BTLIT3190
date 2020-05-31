import math
import numpy as np
from matplotlib import pyplot as plt
from get_dict import get_dict

Xtrain = '../datasets/train'
Xtest = '../datasets/test'

totalMailCount = 0 # Đếm tổng số lượng email
mailCount = {} # Đếm số lượng email theo từng nhãn
wordCounts = {} # Đếm xem có bao nhiêu từ trong từng nhãn
totalFreqWord = {} # Đếm số lần xuất hiện của từng từ xuất hiện trong nhãn
prior = {} # Tính P(type)
conditional = {} # Tính P(tj|type)
dictionary = set()

def set_dict():
    global dictionary

    dicter = get_dict()

    for index, k in enumerate(dicter):
        if index > 6000:
            break
        dictionary.add(k[0])


def BinomialClassifier():
    global dictionary
    global totalMailCount
    global mailCount
    global wordCounts
    global prior
    global conditional
    global totalFreqWord

    wordCounts.setdefault("spam", {})
    wordCounts.setdefault("ham", {})

    totalFreqWord.setdefault("spam", {})
    totalFreqWord.setdefault("ham", {})

    for word in dictionary:
        wordCounts["spam"].setdefault(word, 0)
        wordCounts["ham"].setdefault(word, 0)
        totalFreqWord["spam"].setdefault(word, 0)
        totalFreqWord["ham"].setdefault(word, 0)
        
    trainer = open(Xtrain, 'r')

    for line in trainer:
        totalMailCount += 1
        tokens = line.split()
        type = tokens[1]
        mailCount.setdefault(type, 0)
        mailCount[type] += 1

        for i in range(2, len(tokens), 2):
            if tokens[i] in dictionary:
                wordCounts[type][tokens[i]] += 1
                totalFreqWord[type][tokens[i]] += int(tokens[i+1])


    for type, count in mailCount.items():
        prior.setdefault(type, 0.0)
        prior[type] = count/totalMailCount

    conditional.setdefault("spam", {})
    conditional.setdefault("ham", {})

    for type, attribute in totalFreqWord.items():
        negTYPE = "spam" if type == "ham" else "ham"
        for word, count in attribute.items():
            conditional[type].setdefault(word, 0)
            a = (totalFreqWord[type][word] * wordCounts[type][word] + 1)/(mailCount[type] + 1)
            b = (totalFreqWord[negTYPE][word] * wordCounts[negTYPE][word] + 1)/(mailCount[negTYPE] + 1)
            conditional[type][word] = a/(a+b)


def performClassification():
    global conditional

    testData = open(Xtest,'r')

    actualClassList = list()
    calculatedClassList = list()

    for eachLine in testData:
        tokens = eachLine.split()
        actualClass = tokens[1]

        predictedClass = ""
        wordList = []
        for index, word in enumerate(tokens[2:]):
            if index % 2 == 0:
                wordList += [word]

        pHam = math.log10(prior["ham"]*17)
        pSpam = math.log10(prior["spam"])

        for word in wordList:
            if word in dictionary:
                pHam += math.log10(conditional["ham"][word])
                pSpam += math.log10(conditional["spam"][word])

        if pSpam > pHam:
            predictedClass = "spam"
        else:
            predictedClass = "ham"

        actualClassList.append(actualClass)
        calculatedClassList.append(predictedClass)

    testData.close()

    messageBox(actualClassList, calculatedClassList)


def messageBox(groundTruthList, predictedTruthList):
    correctSpam = 0 # TP: Thư rác vào spam
    correctHam = 0 # TN: Thư thường vào inbox
    incorrectSpam = 0 # FP: Thư thường vào spam
    incorrectHam = 0 # FN: Thư rác vào inbox

    for actual, predicted in zip(groundTruthList, predictedTruthList):
        if predicted in "spam":
            if actual in "spam":
                correctSpam += 1
            elif actual in "ham":
                incorrectSpam += 1
                
        elif predicted in "ham":
            if actual in "ham":
                correctHam += 1
            elif actual in "spam":
                incorrectHam += 1

    accuracy = 100 * (correctHam+correctSpam)/(correctHam+correctSpam+incorrectHam+incorrectSpam)
    precision = 100 * correctSpam/(correctSpam+incorrectSpam)
    recall = 100 * correctSpam/(correctSpam+incorrectHam)
    fmeasure = 2 * precision*recall/(precision+recall)

    confusion_matrix = np.array([[correctSpam, incorrectHam],
                                [incorrectSpam, correctHam]])
    print("Confusion Matrix:\n", confusion_matrix)
    print("\nAccuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}\n".format(accuracy, precision, recall, fmeasure))


if __name__ == "__main__":
    set_dict()
    BinomialClassifier()
    performClassification()
