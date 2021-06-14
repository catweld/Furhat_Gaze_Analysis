import os
import codecs
import datetime
import matplotlib.pyplot as plt
import pandas as pd

def readWords(file_name):
    parsedLines = openAndReadLogFile(file_name, ['Log_Type.VOICE_ACTIVATION', "Log_Type.GAME_WORD"])
    timing = {}
    for elem in parsedLines:
        if elem[2].find("GAME_WORD") >= 0:
            currWord = elem[3]
            currSecs = float(elem[1])
            #if len(timing.keys()) > 0:
                #print(currSecs - list(timing.keys())[-1])
            timing[currSecs] = currWord
    return timing


# TODO, change for adding VADs to game words
def joinWordsAndVAD(file_name, timing):
    parsedLines = openAndReadLogFile(file_name, ['Log_Type.VOICE_ACTIVATION', "Log_Type.GAME_WORD","Log_Type.CONFIG"])
    parseddf = pd.DataFrame(parsedLines)
    # newlist = sorted(list(timing.keys()), key=lambda k: timing[k])
    currIndex = 0
    for elem in parsedLines:
        if elem[2].find("ACT") >= 0:
            currSecs = float(elem[1])
            timing[currSecs] = [elem[3], float(elem[5])]
    return timing


def openAndReadLogFile(file_name, types):
    listResult = []
    with codecs.open(file_name, 'r', 'utf-8') as f:
        # Use the first line to get names for columns
        for line in f:
            if line.find("Log_Type.") > 0:
                this_line = line.split("\t")

                for type_elem in types:
                    # print(this_line[1][1:], type_elem, type_elem.find(this_line[1]))
                    compare_string = this_line[2]
                    if compare_string == type_elem:
                        # if this_line[1] in types:
                        # print(this_line)
                        listResult.append([word.replace('\n', '').replace('\'', '') for word in this_line])

    return listResult

if __name__ == '__main__':
    pair_of_files = [
        ["Sep-23-2020_12-55-50_control_046_Guesser.log","Sep-23-2020_12-56-15_control_046_VADGaze.log"]
    ]
    path = os.path.join(os.getcwd(), "../logs_Sep-23-2020")
    # print(path)
    # resultDict = {}
    # for file in os.listdir(path):
    #    if file.find("Guesser")!=-1:
    #        resultDict = readRecognizedWordsPerGuess(os.path.join(path, file), resultDict)
    listSums = []
    list_x = []
    list_y = []
    for files in pair_of_files:
        timing = readWords(os.path.join(path, files[0]))
        timing = joinWordsAndVAD(os.path.join(path, files[1]), timing)
        # print(timing)
        # compute time spoken per word
        dictSpoken = {'left': 0, 'right': 0}
        newList = sorted(list(timing.keys()))
        start_time = newList[0]
        resultDict = {}
        curr_word = ""
        for i in range(len(newList)):
            if type(timing[newList[i]]) != str:
                if curr_word != "":
                    if resultDict[curr_word][timing[newList[i]][0]] == 0:
                        resultDict[curr_word][timing[newList[i]][0]] = -timing[newList[i]][1]
                        print(timing[newList[i]])
                    else:
                        dictSpoken[timing[newList[i]][0]] = timing[newList[i]][1]
                else:
                    # do nothing if they talk before first word
                    pass
            else:
                if dictSpoken['left'] != 0 or dictSpoken['right'] != 0:
                    print("Start: ", resultDict[curr_word])
                    resultDict[curr_word]['left'] += dictSpoken['left']
                    resultDict[curr_word]['right'] += dictSpoken['right']
                print("End: ", dictSpoken, "\n Between: ", resultDict)
                dictSpoken = {'left': 0, 'right': 0}
                curr_word = timing[newList[i]]
                resultDict[curr_word] = {'left': 0, 'right': 0}
                # print(curr_word)

        for elem in resultDict.keys():
            print("left: {} right: {}, total: {}".format(resultDict[elem]['left'], resultDict[elem]['right'],
                                                         resultDict[elem]['left'] + resultDict[elem]['right']))
            listSums.append(resultDict[elem]['left'] + resultDict[elem]['right'])

        for word in resultDict.keys():
            list_x.append(list(timing.keys())[list(timing.values()).index(word)] - start_time)
            list_y.append(resultDict[word]['left'] + resultDict[word]['right'])

        del listSums[-1]
        del list_x[-1]
        del list_y[-1]
    import statistics

    print(statistics.mean(listSums), statistics.stdev(listSums))
    import numpy as np

    hist = np.histogram(np.array(listSums), bins=4)
    print(hist[0] / np.sum(hist[0]))
    print(hist)
    import matplotlib.pyplot as plt

    plt.hist(listSums, bins=4)
    plt.show()
    hist = np.histogram(np.array(listSums), bins=5)
    print(hist[0] / np.sum(hist[0]))
    print(hist)

    plt.hist(listSums, bins=5)

    # plot time to time

    plt.plot(list_x, list_y)
    plt.savefig("plot.png")