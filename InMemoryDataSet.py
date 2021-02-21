from scipy.sparse import lil_matrix
import numpy as np
import random

class InMemoryDataSet:
    def __init__(self, filename):
        activityOptions = ["study", "read", "rest", "sleep", "laptop/TV", "sport", "house-activities",
                           "friends-night-at-home"]

        try:
            with open(filename, 'r') as fileDescriptor:
                fileLines = fileDescriptor.readlines()

                noOfTokensOnLine = fileLines[0].split(' ')

                maxRowIndex = len(fileLines)
                maxColIndex = len(noOfTokensOnLine) - 1
                self.featureMatrix = lil_matrix((maxRowIndex, maxColIndex), dtype=np.int)

                self.verdicts = []

                index = 0
                for currentLine in fileLines:
                    lineToAddInFeatureMatrix = []

                    tokens = currentLine.split(' ')

                    for token in tokens[:-1]:
                        if str(token) in activityOptions:
                            token = activityOptions.index(token)
                        lineToAddInFeatureMatrix.append(token)

                    lineToAddInFeatureMatrix = np.array(lineToAddInFeatureMatrix)
                    self.featureMatrix[index] = lineToAddInFeatureMatrix
                    index += 1

                    valueToAddInVerdictVector = tokens[len(tokens) - 1]

                    self.verdicts.append(valueToAddInVerdictVector)

        except FileNotFoundError as ex:
            print("Error: File not found! Details: " + str(ex))

    def shuffle_samples(self):
        indices = np.arrange(self.featureMatrix.shape[0])
        random.shuffle(indices)
        self.featureMatrix = self.featureMatrix[indices]
        self.verdicts = self.verdicts[indices]

    def getFeatureMatrix(self):
        return self.featureMatrix

    def getVerdicts(self):
        return self.verdicts

    def __str__(self):
        toReturnString = ""

        toReturnString += "Matrix: \n"
        for row in self.featureMatrix:
            for token in row:
                toReturnString += str(token) + " "
            toReturnString += "\n"

        toReturnString += "Vector: \n"
        for verdict in self.verdicts:
            toReturnString += str(verdict) + " "

        return toReturnString
