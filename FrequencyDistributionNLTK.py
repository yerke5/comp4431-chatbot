import nltk
import spacy

def readText(fileName):
    file = open(fileName, "r")
    content = file.read()
    file.close()
    return content

def getFreqs(fileName, numWords):
    content = readText(fileName)
    tokens = nltk.word_tokenize(content)
    frequency_dist = nltk.FreqDist(tokens)
    frequency_dist.plot(numWords)

def getFacts(fileName):
    content = readText(fileName)

    ### Load spaCy's English NLP model
    nlp = spacy.load('en_core_web_lg')

    ### Parse the text with spaCy
    ### Our 'document' variable now contains a parsed version of text.
    document = nlp(content)

    ### print out all the named entities that were detected
    for entity in document.ents:
        print(entity.text, entity.label_)

getFacts("text.txt")
getFreqs("text.txt", 50)
getFreqs("history.txt", 50)
