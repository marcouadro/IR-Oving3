import random
import codecs
import nltk
import string
import gensim
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

# Functions:
def makeParagraphs(file):
    paragraphsList = []
    paragraph = ''

    for line in f.readlines():
        # if it is not possible to strip the line, it means it contains no words
        # and is therefore the end of a paragraph
        if not line.strip():
            # simple check to see that the paragraph is not empty.
            if (paragraph != ""):
                paragraphsList.append(paragraph)
                paragraph = ""
        else:
            paragraph += line

    return paragraphsList


def removeWord(word, array):
    '''
    Removes a paragraph from the array
    should it contain the given word.
    '''

    tempArray = []
    for item in array:
        if word not in item:
            tempArray.append(item)

    return tempArray


def tokenize(array):
    tempArray = []
    for item in array:
        tempArray.append(nltk.word_tokenize(item))
    return tempArray


def removePunctuationAndLower(array):
    remove = string.punctuation + "\n\r\t"  # The characters we want removed
    tempArray = []
    tempParagraphsList = []
    # Checking if array is 2D or 1D so it works on both
    if isinstance(array[0], list):
        for i in range(len(array)):
            for j in range(len(array[i])):
                word = array[i][j]
                newWord = ''
                for char in word:
                    if char not in remove:
                        newWord += char
                if newWord != '':
                    tempParagraphsList.append(newWord.lower())

            tempArray.append(tempParagraphsList)
            tempParagraphsList = []
    else:
        for i in range(len(array)):
            word = array[i]
            newWord = ''
            for char in word:
                if char not in remove:
                    newWord += char
            if newWord != '':
                tempArray.append(newWord.lower())
    return tempArray


def stem(array):
    stemmer = PorterStemmer()
    if isinstance(array[0], list):
        for i in range(len(array)):
            for j in range(len(array[i])):
                array[i][j] = stemmer.stem(array[i][j])
    else:
        for i in range(len(array)):
            array[i] = stemmer.stem(array[i])

    return array


def getStopWordIDs(stopwords, dictionary):
    stopWordIDs = []
    for word in stopwords:
        try:
            stopWordIDs.append(dictionary.token2id[word])
        except:
            pass
    return stopWordIDs


def convertToBagOfWords(array, dictionary):
    vector = []
    for p in array:
        vector.append(dictionary.doc2bow(p))
    return vector


def makeQuery(query):
    query = query.split(" ")
    query = removePunctuationAndLower(query)
    query = stem(query)

    return query


# ASSIGNMENT

# -----------1.0------------#

# 1.0 Fix random number generator
random.seed(123)

# 1.1 Open the file with utf-8 encoding
f = codecs.open('pg3300.txt', "r", "utf-8")

# 1.2 Dividing the file into paragraphs
paragraphs = makeParagraphs(f)

# 1.3 Removing paragraphs with the word "Gutenberg"
removeGutenberg = removeWord("Gutenberg", paragraphs)

# 1.4 Tokenizing the paragraphs
tokenized = tokenize(removeGutenberg)

# 1.5 Removing punctuation from the array and also "\n\t\r"
noPunctuation = removePunctuationAndLower(tokenized)

# 1.6 Creating a stemmed array:
stemmedArray = stem(noPunctuation)

# -----------2.0------------#

# 2.0 Create dictionary based on our stemmed list
dictionary = gensim.corpora.Dictionary(noPunctuation)

# 2.1 Get the IDs for the stop words in our dictionary
stopFile = codecs.open("stopwords.txt", "r", "utf-8").read().split(",")

# Removing punctuation and whitespace
stopFile = removePunctuationAndLower(stopFile)

# Getting the IDs for the stopwords from the dictionary
stopWordIDArray = getStopWordIDs(stopFile, dictionary)

# 2.1 Filter out the stopwords from our dictionary
dictionary.filter_tokens(stopWordIDArray)

# 2.2 Converting our list to a bag of words (corpus).
corpus = convertToBagOfWords(noPunctuation, dictionary)

# -----------3.0------------#

# 3.1 Building a TD-IDF model using corpus
tfidf_model = gensim.models.TfidfModel(corpus)

# 3.2 Mapping corpus in to TF-IDF weights
tfidf_corpus = tfidf_model[corpus]

# 3.3 Constructing a MatrixSimilarity object
matrixSimimlarity = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# 3.4 Repeat the above procedure for LSI model using as an input the corpus with TF-IDF weights.
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)

lsi_corpus = lsi_model[tfidf_corpus]

lsi_matrix = gensim.similarities.MatrixSimilarity(lsi_corpus)

# 3.5
print("The first 3 LSI topics are:\n" + str(lsi_model.show_topics(3, 3)) + "\n")

# -----------4.0------------#

# 4.1
queryToCheck = "How taxes influence Economics?"
query = makeQuery(queryToCheck)
query = dictionary.doc2bow(query)

# 4.2
query_tfidf = tfidf_model[query]
print("For the query: '" + queryToCheck + "' The TF-IDF weights are:")
print(str(dictionary.get(query_tfidf[0][0])) + ": " + str(query_tfidf[0][1]) + ", " + str(
    dictionary.get(query_tfidf[1][0])) + ": " + str(query_tfidf[1][1]) + ", " + str(
    dictionary.get(query_tfidf[2][0])) + ": " + str(query_tfidf[2][1]))

# 4.3
print("\nThe top 3 paragraphs based on the weights are:")

docSim = enumerate(matrixSimimlarity[query_tfidf])
docs = sorted(docSim, key=lambda kv: -kv[1])[:3]

for doc in docs:
    text = "\n[Paragraph {0}]\n {1}".format(doc[0], paragraphs[doc[0]])
    print(text)

# 4.4
print("\nFor the query: '" + queryToCheck + "' The top 3 topics are:")

lsi_query = lsi_model[query_tfidf]
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]

for i in range(len(topics)):
    text = "\n[Topic {0}]\n {1}".format(topics[i][0], lsi_model.print_topic(topics[i][0]))
    print(text)

print("\nAnd the top 3 paragraphs according to the LSI model are:")
docSim = enumerate(lsi_matrix[lsi_query])
docs = sorted(docSim, key=lambda kv: -kv[1])[:3]

for doc in docs:
    text = "\n[Paragraph {0}]\n {1}".format(doc[0], paragraphs[doc[0]])
    print(text)