import random
from InMemoryDataSet import InMemoryDataSet
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
   print("main")

def constructDatasetPhase3():
    file = open("datasetPhase2.txt","r")
    entriesPhase2 = file.readlines()

    fileToWrite = open("datasetPhase3.txt", "a")

    for entryPhase2 in entriesPhase2:
        kelvinIntervalValues = getKelvinValueInterval(entryPhase2)

        kelvinValue = random.randint(kelvinIntervalValues[0], kelvinIntervalValues[1])

        entryPhase3 = constructEntryPhase3(entryPhase2, kelvinValue)

        try:
            writeToFile(entryPhase3, fileToWrite)
        except IOError:
            print("Unable to write to file phase 3!")

    fileToWrite.close()
    file.close()

def constructEntryPhase3(entry, kelvinTemperature):
    return str(entry.rstrip("\n")) + " " + str(kelvinTemperature)

def getKelvinValueInterval(entry):
    entryCharacteristics = entry.split(" ")
    kelvinTemperature = []

    activityType = entryCharacteristics[4].rstrip("\n")

    if activityType == "study":
        kelvinTemperature.append(4000)
        kelvinTemperature.append(5200)
    if activityType == "read":
        kelvinTemperature.append(2700)
        kelvinTemperature.append(3000)
    if activityType == "rest":
        kelvinTemperature.append(2500)
        kelvinTemperature.append(3500)
    if activityType == "sleep":
        kelvinTemperature.append(1000)
        kelvinTemperature.append(2000)
    if activityType == "laptop/TV":
        kelvinTemperature.append(3000)
        kelvinTemperature.append(3500)
    if activityType == "sport":
        kelvinTemperature.append(5000)
        kelvinTemperature.append(6500)
    if activityType == "house-activities":
        kelvinTemperature.append(3000)
        kelvinTemperature.append(3600)
    if activityType == "friends-night-at-home":
        kelvinTemperature.append(2500)
        kelvinTemperature.append(3200)

    eyeDiseases = str(entryCharacteristics[3])

    if eyeDiseases == "True":
        if kelvinTemperature[1] <= 4500:
            kelvinTemperature[1] += 100
            kelvinTemperature[0] += 100
        elif kelvinTemperature[1] > 4500:
            kelvinTemperature[1] -= 200
            kelvinTemperature[0] -= 200

    hoursSinceAwake = int(str(entryCharacteristics[0]))

    if hoursSinceAwake > 8 and hoursSinceAwake < 16:
        if kelvinTemperature[1] <= 3000:
            kelvinTemperature[1] -= 10 * (hoursSinceAwake - 8)
            kelvinTemperature[0] -= 10 * (hoursSinceAwake - 8)
        elif kelvinTemperature[1] > 3000 and kelvinTemperature[1] <= 4500:
            kelvinTemperature[1] -= 5 * (hoursSinceAwake - 8)
            kelvinTemperature[0] -= 5 * (hoursSinceAwake - 8)
        elif kelvinTemperature[1] > 4500:
            kelvinTemperature[0] += 10 * (hoursSinceAwake - 8)
    if hoursSinceAwake >= 16:
        if kelvinTemperature[1] <= 3000:
            kelvinTemperature[1] -= 30 * (hoursSinceAwake - 8)
            kelvinTemperature[0] -= 30 * (hoursSinceAwake - 8)
        elif kelvinTemperature[1] > 3000 and kelvinTemperature[1] <= 4500:
            kelvinTemperature[1] += 10 * (hoursSinceAwake - 8)
            kelvinTemperature[0] += 10 * (hoursSinceAwake - 8)
        elif kelvinTemperature[1] > 4500:
            kelvinTemperature[1] += 15 * (hoursSinceAwake - 8)
            kelvinTemperature[0] += 15 * (hoursSinceAwake - 8)

    hoursUntillSunset = int(str(entryCharacteristics[1]))
    hoursSinceSunrise = int(str(entryCharacteristics[2]))

    if activityType == "study" or activityType == "sport":
        if hoursUntillSunset > 7 or hoursSinceSunrise < 2:
            kelvinTemperature[1] += 1000
            kelvinTemperature[0] += 1000
        elif (hoursUntillSunset < 5) or hoursSinceSunrise >= 7:
            kelvinTemperature[1] += 200
            kelvinTemperature[0] += 200
        elif (hoursUntillSunset >= 5 and hoursUntillSunset <= 7) or (hoursSinceSunrise >= 2 and hoursSinceSunrise < 7):
            kelvinTemperature[1] -= 1000
            kelvinTemperature[0] -= 1000

    elif activityType == "friends-night-at-home" or activityType == "sleep" or activityType == "rest" or activityType == "read":
        if hoursUntillSunset > 7 or hoursSinceSunrise < 2:
            kelvinTemperature[1] -= 200
            kelvinTemperature[0] -= 200
        elif (hoursUntillSunset < 5) or hoursSinceSunrise >= 7:
            kelvinTemperature[1] += 500
            kelvinTemperature[0] += 500
        elif (hoursUntillSunset >= 5 and hoursUntillSunset <= 7) or (hoursSinceSunrise >= 2 and hoursSinceSunrise < 7):
            kelvinTemperature[1] -= 500
            kelvinTemperature[0] -= 500

    elif activityType == "house-activities" or activityType == "laptop/TV":
        if hoursUntillSunset > 7 or hoursSinceSunrise < 2:
            kelvinTemperature[1] += 500
            kelvinTemperature[0] += 500
        elif (hoursUntillSunset < 5) or hoursSinceSunrise < 7:
            kelvinTemperature[1] += 1000
            kelvinTemperature[0] += 1000

    return kelvinTemperature

def constructDatasetPhase2():
    activityOprions = ["study", "read", "rest", "sleep", "laptop/TV", "sport", "house-activities",
                       "friends-night-at-home"]

    file = open("datasetPhase1.txt",
                "r")
    entriesPhase1 = file.readlines()

    fileToWrite = open("datasetPhase2.txt", "a")

    for activity in activityOprions:

        for entryPhase1 in entriesPhase1:
            entryPhase2 = constructEntryPhase2(entryPhase1, activity)

            try:
                writeToFile(entryPhase2, fileToWrite)
            except IOError:
                print("Unable to write to file phase 2!")

    fileToWrite.close()
    file.close()

def constructEntryPhase2(entry, activity):
    return str(entry.rstrip("\n")) + " " + str(activity)

def constructDatasetPhase1():
    entriesToBeWritten = []

    cnt = 0

    file = open("datasetPhase1.txt","a")

    while cnt < 1250:
        cnt = len(entriesToBeWritten)
        entryIsWrittenToFile = False

        entry = generateRandomEntryOfUserPersonsalParams()

        if (isEntryInDataset(entry, entriesToBeWritten) == False):
            try:
                writeToFile(entry, file)
                entryIsWrittenToFile = True
            except IOError:
                print("Unable to write to file phase 1!")

            if (entryIsWrittenToFile):
                entriesToBeWritten.append(entry)

        else:
            print("Entry { " + entry + " } is already in the file!")

    file.close()

def generateRandomEntryOfUserPersonsalParams():
    hours = random.randint(1, 24)
    untillSunset = random.randint(0, 12)
    sinceSunrise = random.randint(0, 12)
    while sinceSunrise + untillSunset < 6:
        untillSunset = random.randint(0, 12)
        sinceSunrise = random.randint(0, 12)
    eyeDiseases = random.choice([1, 0])

    return constructEntryPhase1(hours, untillSunset, sinceSunrise, eyeDiseases)

def constructEntryPhase1(hours, untillSunset, untillSunrise, eyeDiseases):
    return str(hours) + " " + str(untillSunset) + " " + str(untillSunrise) + " " + str(eyeDiseases)

def isEntryInDataset(entry, entries):
    if entry in entries:
        return True
    else:
        return False

def writeToFile(entry, file):
    file.write("\n")
    file.write(entry)

main()
