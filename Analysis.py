import nltk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
import re
# some code for getSentiment was adapted from
# https://dev.to/rodolfoferro/sentiment-analysis-on-trumpss-tweets-using-python-

def clean(string):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", string).split())

def analyzeSentiment(string):
    analysis = TextBlob(clean(string))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

def readText(fileName):
    file = open(fileName, "r")
    content = file.read()
    file.close()
    return content

def getFreqs(fileName, numWords, plot=False):
    content = readText(fileName)
    tokens = nltk.word_tokenize(content)
    frequency_dist = nltk.FreqDist(tokens)
    if plot:
        frequency_dist.plot(numWords)

def getAvgNumSnts(fileName):
    file = open(fileName, "r")
    avg = 0.0
    c = 1
    lines = file.readlines()
    for line in lines:
        if(c % 2 == 0):
            avg += len(line.split("."))
        c += 1
    numLines = len(lines) / 2.0
    print(numLines)
    print(avg)
    return avg / numLines
    
def getAvgWordsPerQuestion(fileName):
    file = open(fileName, "r")
    avg = 0
    c = 1
    lines = file.readlines()
    for line in lines:
        if(c % 2 == 0):
            avg += len(line.split(" "))
        c += 1
    return avg / (len(lines) / 2)

def getFacts(fileName):
    content = readText(fileName)

    # get spacy model
    nlp = spacy.load('en_core_web_lg')

    # parse text
    document = nlp(content)

    # print entities
    for entity in document.ents:
        print(entity.text, entity.label_)

def getSentiment(fileName, plotFile="sentiment.png"):
    file = open(fileName, "r")
    lines = file.readlines()
    c = 1
    data = []
    for line in lines:
        if(c % 2 == 0):
            data.append(analyzeSentiment(line))
        c += 1
    data = np.asarray(data)
    df = pd.DataFrame(data)
    df.columns = ["Sentiment"]

    pos = 0
    neg = 0
    neut = 0
    for index, row in df.iterrows():
        if row["Sentiment"] == 1:
            pos += 1
        elif row["Sentiment"] == 0:
            neut += 1
        else:
            neg += 1
    print("The number of positive responses:", pos)
    print("The number of negative responses:", neg)
    print("The number of neutral responses:", neut)
    labels = 'Positive', 'Negative', 'Neutral'
    sizes = [pos, neg, neut]
    colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0)
     
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title("Sentiment Distribution")
    plt.savefig(plotFile)
    #plt.show()

article = "text.txt"
history = "history.txt"
print("---------------------------")
print("FACTS")
print("---------------------------")
getFacts(article)
print("---------------------------")
print("AVERAGES")
print("---------------------------")
print("The average number of sentences per answer is", getAvgNumSnts(history))
print("The average number of words per answer is", getAvgWordsPerQuestion(history))
print("---------------------------")
print("SENTIMENT ANALYSIS")
print("---------------------------")
getSentiment(history)
print("The sentiment pie chart has been saved successfully to sentiment.png")
print("---------------------------")
print("FREQUENCIES")
print("---------------------------")
getFreqs(article, 50, plot=True)
getFreqs(history, 50, plot=True)
