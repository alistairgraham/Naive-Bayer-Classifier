import sys


def loadData(filename):
    falseCount = 0
    trueCount = 0
    trainingList = []

    with open(filename) as file:
        for line in file.read().splitlines():
            tokenListUnformatted = line.split()
            tokenList = []
            for s in tokenListUnformatted:
                featureValue = int(s)
                if (not(featureValue == 0 or featureValue == 1)):
                    raise ValueError("Expected 1 or 0")
                tokenList.append(int(s))

            trainingList.append(tokenList)

            if (tokenList[len(tokenList)-1] == 0):
                falseCount += 1
            elif (tokenList[len(tokenList)-1] == 1):
                trueCount += 1
            else:
                raise ValueError("Expected 1 or 0")

    return trainingList, falseCount, trueCount


def computeProbalityTable(trainingList):
    countTable = []
    dataLength = len(trainingList[0])
    # Initialise with 1's to prevent dividing by 0
    for i in range(0, ((dataLength-1)*2)+1):
        countTable.append([1,1])
    # count up features
    for email in trainingList:
        for i in range(0, dataLength):
            # if class feature
            if (i == dataLength-1):
                countTable[i*2][email[i]] += 1
                continue
            countTable[(i*2)+email[dataLength-1]][email[i]] += 1
    #work out probabilities
    probabilityTable = []
    for feature in countTable:
        total = feature[0] + feature[1]
        probabilityTable.append([feature[0]/total, feature[1]/total])

    return probabilityTable


def classifyUnlabelled(testList, probabilityTable):
    predictedList = []
    for email in testList:
        probFalse = calculateProbability(email, probabilityTable, 0)
        probTrue = calculateProbability(email, probabilityTable, 1)
        if (probFalse > probTrue):
            predictedList.append(0)
        else:
            predictedList.append(1)

    return predictedList


def calculateProbability(email, probabilityTable, classNum):
    probability = 1
    for i in range(0, len(email)):
        probability *= probabilityTable[(i*2)+classNum][email[i]]
    probability *= probabilityTable[len(probabilityTable)-1][classNum]

    return probability

def outputFeatureProbabilities(probTable):
    spamFeatureProb = []
    for i in range(0, len(probTable)-1):
        spamFeatureProb.append(probTable[i][1])
    print(spamFeatureProb)
    return


def main():
    trainingList, falseCount, trueCount = loadData("resources/"+sys.argv[1])
    probabilityTable = computeProbalityTable(trainingList)
    testList,x, y = loadData("resources/"+sys.argv[2])
    predictedList = classifyUnlabelled(testList, probabilityTable)
    outputFeatureProbabilities(probabilityTable)
    print(predictedList)
    return


main()
