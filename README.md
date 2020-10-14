##### -*- coding: utf-8 -*-
"""
Created on Thu Sep  23 12:20:58 2018

**@autor instabilotheque, tim a.

**@title: Babylonische Sorgen - Oulipotische Balladen aus Suchmaschinen-Anfragen in Python**

"""
#### "Alle festen Formen sind per Definition oulipotisch. Oulipotischer als das Sonett geht es kaum. Wir empfehlen daher auch die **Ballade** , das kˆnigliche Lied, das Rondeau, das Pantoum usw. (feste Form, falls vorhanden)."

#### *Quelle: http://oulipo.net/fr/contraintes/formes-fixes


import csv
import itertools
import io
from os import path
import MySQLdb
import pygermanet
import re
import random
from nltk.tokenize import TweetTokenizer
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
from collections import Counter
from os import path
import matplotlib.pyplot as plt
import six

def getDataSearchQueries():                              
    
    db = MySQLdb.connect(
    host="",
    user="",
    passwd="",
    db=""
    )
    
    cursor = db.cursor(MySQLdb.cursors.DictCursor)
    cursor2 = db.cursor(MySQLdb.cursors.DictCursor)
    #cursor3 = db.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT DISTINCT results FROM jesuisautocomplete_persoenlich WHERE Length(results) > 20 ORDER BY RAND()")
    cursor2.execute("SELECT DISTINCT results FROM jesuisautocomplete WHERE Length(results) > 20 ORDER BY RAND()")
    return cursor.fetchall() + cursor2.fetchall()


def isprintable(s, codec='utf8'):
    try: s.decode(codec)
    except UnicodeDecodeError: return False
    else: return True

def getDataBallad():                              
    
    db = MySQLdb.connect(
    host="",
    user="",
    passwd="",
    db=""
    )
    
    cursor3 = db.cursor(MySQLdb.cursors.DictCursor)
    cursor3.execute("SELECT * FROM balladen WHERE title='erlkoenig' ;")
    return cursor3.fetchall()

def makeQuery(keys):

    query_result = []
    
    word_set = set(keys)

    for member in queryData:
        phrase_set = set(member.split())
        if word_set.intersection(phrase_set):
            query_result.append(member)
    return query_result

def databaseToArray():
    queryData = []
    for k in getDataSearchQueries():
        if isprintable(k['results']) == False:
            pass
        else:
            queryData.append(k['results'])
    return queryData
    
def tokenizeBallad():
    flattenedBallad = []
    for b1 in getDataBallad():
        flattenedBallad.append(b1['results'])
    return flattenedBallad

tknzr = TweetTokenizer()

def stripWords(arrayoftokenizedstuff):
    temp = []
    strippedarray = []
    for token in arrayoftokenizedstuff:
        if len(token) < 6 or type(token) == str:
            pass
        else:
            strippedarray.append(((''.join(re.findall("[a-zA-Z]+", token)).lower())))
    return strippedarray
        
def tokenizeResults(arrayofresults):
    tokenizedresults = []
    temp1 = []
    for sentence in arrayofresults:
        temp1.append(tknzr.tokenize(sentence))
    l_autocompleted = list(itertools.chain.from_iterable(temp1))

    str_autocompleted = u' '.join(l_autocompleted)

    tokenizedresults = str_autocompleted.split()
    return tokenizedresults 

def filterResults(tokenizedresults):
    german_stopwords =  set(stopwords.words("german")) # NLTK LIBRARY STOPWORDS
    english_stopwords = set(stopwords.words("english")) # NLTK LIBRARY STOPWORDS
    arr_custom_stopwords = []
    fltrd_results = []
    with open("custom_stopwords_new.csv") as csvfile:
        reader2 = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) 
        for row2 in reader2:
            arr_custom_stopwords.append(row2)
    custom_stopwords =set(itertools.chain.from_iterable(arr_custom_stopwords))
    all_stopwords = german_stopwords | english_stopwords | custom_stopwords
    for w in tokenizedresults:
        if w not in all_stopwords:
            fltrd_results.append(w)

    frequency = FreqDist(fltrd_results)
    return frequency

def extractString(tokenizedBallad, stringOperation):
    firstWords = []
    for w in tokenizedBallad:
        firstWords.append(eval(stringOperation))
    return firstWords

def similarTokens(keystring):
    suggestions = []
    listofsimilar = []
    for z in queryData:
        suggestions.append(z)
        
    for matching in suggestions:
        if (keystring == matching.split(' ', 1)[0] ):
            listofsimilar.append(matching)
    return listofsimilar

def matchTokens(extractedString):
    matchingMap = [] 
    for eachToken in extractedString:
        if len(similarTokens(eachToken)) > 0:
            lines = similarTokens(eachToken)
            randomint = random.randint(0,(len(similarTokens(eachToken)) -1))
            matchingMap.append(lines[randomint])
        else:
            matchingMap.append(['Not Found'])
    return matchingMap

def formatiere(array):
    temp = []
    formatted = []
    for element in array:
        temp.append(''.join(element))
    
    formatted=list(map(str.strip,temp))  
    return formatted

def sortiere(array):
    formatted = []
    for element in array:
        formatted.append(''.join(sorted(element)))
    return formatted

def formatBallad(whattoprint, linesperverse):
    for index, item in enumerate(whattoprint):
        if (index % linesperverse == 0):
            print('\n')
            print(item)
        else:
            print(item)
    return

def getSynonyms(suchwort):
    syn = germanet.synsets(suchwort)
    synonyme = []
    temp1 = []
    temp2 = []
    temp3 = []
    for element in syn:
        temp1.append(repr((element)))
        temp1.append(repr((element.lemmas)))
        temp1.append(repr((element.rels)))
        temp1.append(repr((element.causes)))
        temp1.append(repr((element.related_tos)))
        temp1.append(repr((element.entails)))
        temp1.append(repr((element.entailed_bys)))
        temp1.append(repr((element.hypernyms)))
        temp1.append(repr((element.hyponyms)))
        temp1.append(repr((element.member_holonyms)))
        temp1.append(repr((element.member_meronyms)))
        temp1.append(repr((element.component_holonyms)))    

    for elementt in temp1:
         temp2.append(re.sub(r'\..*$', "", elementt))
    for elementtt in temp2:
         temp3.append(re.findall(r'(?<=\().*$', elementtt))
    for elementttt in temp3:
        if len(''.join(elementttt)) == 0:
            pass
        else:
            synonyme.append(''.join(elementttt).lower()) 
    synonyme = list(set(synonyme))
    return synonyme

def fileToDict(filepath):                         
    dictA = defaultdict(dict)
    with open(filepath) as fileA:
        for line in fileA:
            words = line.split()                    
            dictA[words[0]] = tuple(words[1:])
    return dictA

STRESSES = {'EEH','EHH','EH','IIH', 'OOH', 'AAH', 'UUH','UU','OO','IH','EX','CX', 'PF', 'RR', 'X','A','O','OE','YO','YR'} 
PHODICT = fileToDict("phodict_ger_2.dic")             

def isSubList(listA, listB):                    # Checkt ob Liste in anderer Liste enthalten ist. Verhindert Pseudoreime
    if len(listA) < len(listB):
        return isSubList(listB, listA)
    n = len(listB)
    for start in range(len(listA)-n+1):
        if all(listA[start+i] == listB[i] for i in range(n)):
            return True
    return False

def isRhyme(wordA, wordB):
    soundsA, soundsB = PHODICT[wordA], PHODICT[wordB]  
    p1 = []
    p2 = []
    if isSubList(soundsA, soundsB):
        return False
    for index,sound in enumerate(reversed(soundsA)):
        if sound in STRESSES:
            break
    return p1[-index-1:] == p2[-index-1:]

def syllableCount(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count

def syllablephraseCount(phrase):
    sumofsyllables = []
    words = [word for line in lines for word in phrase.split()]
    for k in words:
        sumofsyllables.append(syllableCount(k))
    return sum(sumofsyllables)

def makeSyllableMap():         
    dictA = {i: set() for i in range(0,15)}     # Gibt ein Dictionary zur¸cK, Key: Anzahl der Silben, Value: Tokens  
    for word in PHODICT.keys():
        dictA[syllableCount(word)].add(word)
    return dictA

SYLDICT = makeSyllableMap()

def getRhymes(word):                             # Findet Kandidaten f¸r Reime
    sounds = PHODICT[word]
    if len(sounds) == 0:
        yield 'not found'
    else:
        ending = []
        for index,sound in enumerate(reversed(sounds)):
            if sound in STRESSES:
                ending = sounds[-index-1:]
                break
        yielded = set()
        for wordB, soundsB in PHODICT.items():
            if (ending == soundsB[-index-1:]) and (soundsB not in yielded) and (not isSubList(sounds, soundsB)):
                yielded.add(soundsB)
                yield wordB

def hasRhymes(word):
    if list(getRhymes(word)) == 0:
        return False
    else:
        return True

def readSentLexNeg():
    establish_konflikt_sentlex_neg_temp = []
    establish_konflikt_sentlex_neg = []
    sentlex_neg = []
    txt2 = io.open('sentlex_neg.txt').read()
    sent_neg = txt2.split()
    for sentiment in sent_neg:
        establish_konflikt_sentlex_neg.append(re.split(r',(?![^[]*\])', sentiment))
    temp_autocompleted = list(itertools.chain.from_iterable(establish_konflikt_sentlex_neg))
    sentlex_neg = stripWords(tokenizeResults(temp_autocompleted))
    return sentlex_neg
    
def readSentLexPos():
    establish_konflikt_sentlex_neg_temp = []
    establish_konflikt_sentlex_neg = []
    sentlex_neg = []
    txt2 = io.open('sentlex_pos.txt').read()
    sent_neg = txt2.split()
    for sentiment in sent_neg:
        establish_konflikt_sentlex_neg.append(re.split(r',(?![^[]*\])', sentiment))
    temp_autocompleted = list(itertools.chain.from_iterable(establish_konflikt_sentlex_neg))
    sentlex_neg = stripWords(tokenizeResults(temp_autocompleted))
    return sentlex_neg
    
def findWord(n):
    return random.sample(SYLDICT[n], 1)[0]       # Zufallswort aus dem Phonetischen Wˆrterbuch

def makeLimerick():                              # Reim Pattern: 10A,10A,6B,6B,10A 
    e1 = findWord(1)                             
    r1 = tuple(getRhymes(e1))
    e2 = random.choice(r1)
    e5 = random.choice(r1)
    e3 = findWord(1)
    r3 = tuple(getRhymes(e3))
    e4 = random.choice(r3)
    lines = [[findWord(4), findWord(4), e1],
             [findWord(4), findWord(4), e2],
             [findWord(3), e3],
             [findWord(3), e4],
             [findWord(4), findWord(4), e5]]
    return '\n'.join(' '.join(line) for line in lines)


def select(kandidates):
    return random.choice(kandidates)

germanet = pygermanet.load_germanet()

queryData = databaseToArray()

def makeKuckucksEi():
    matchedtokens = []
    firstWordEachLine = 'w.split(" ", 1)[0].lower()'
    tokenizedballad = extractString(tokenizeBallad(), firstWordEachLine )
    matchedtokens = matchTokens(tokenizedballad)
    return matchedtokens

def makeSetting():
    
    # Figuren
    establish_mensch = ['ich', 'mein']
    establish_him = ['ich', 'sie'] 
    establish_her = ['ich', 'er']
    establish_it = ['error', 'interrupt']
    establish_us = ['wir', 'euer', 'eure']

    # Ort
    establish_ort = []
    establish_ort = ['wohnblock','daheim', 'zuhause']
    txt = io.open('staedte.txt').read()
    staedte = txt.splitlines()
    for stadt in staedte:
        establish_ort.append(stadt.encode('utf-8').lower())

    # Intervention
    establish_zuspitzung = ['plˆtzlich', 'jetzt']
    establish_zuspitzung2 = ['sofort', 'dringend']    
    
    # Konflikte
    establish_konflikt_simple = np.array([getSynonyms('teuer'), getSynonyms(u'h‰sslich'),getSynonyms('trennung'), getSynonyms('dick')]) # <- stem this shit
    establish_sentlex_neg = set(readSentLexNeg())

    # Ereignis Transformationen
    establish_transformation = ['damit', 'dadurch', 'deswegen']

    # Gutes
    establish_sentlex_pos = set(readSentLexPos())
    establish_katharsis = ['endlich']


def showStats():
    
    results_establish_mensch = similarTokens(random.choice(establish_mensch))
    r1 = len(results_establish_mensch)
    print('Anzahl gefundener S‰tze f¸r Establish-Mensch: %s ' % r1)
    mc_establish_mensch = filterResults(tokenizeResults(results_establish_mensch)).most_common(20)
    plt.barh(range(len(mc_establish_mensch)),[val[1] for val in mc_establish_mensch], align='center')
    plt.yticks(range(len(mc_establish_mensch)), [val[0] for val in mc_establish_mensch])
    plt.show()
    
    results_establish_sent_lex_neg = makeQuery(establish_sentlex_neg)
    r4 = len(results_establish_sent_lex_neg)
    print('Anzahl gefundener S‰tze f¸r Matches aus dem Sentiment Lexicon Negativ: %s ' % r4)
    mc_establish_sent_lex_neg = filterResults(tokenizeResults(results_establish_sent_lex_neg)).most_common(20)
    plt.barh(range(len(mc_establish_sent_lex_neg)),[val[1] for val in mc_establish_sent_lex_neg], align='center')
    plt.yticks(range(len(mc_establish_sent_lex_neg)), [val[0] for val in mc_establish_sent_lex_neg])
    plt.show()
    
    results_establish_sent_lex_pos = makeQuery(establish_sentlex_pos)
    r5 = len(results_establish_sent_lex_pos)
    print('Anzahl gefundener S‰tze f¸r Matches aus dem Sentiment Lexicon Positiv: %s ' % r5)
    mc_establish_sent_lex_pos = filterResults(tokenizeResults(results_establish_sent_lex_pos)).most_common(20)
    plt.barh(range(len(mc_establish_sent_lex_pos)),[val[1] for val in mc_establish_sent_lex_pos], align='center')
    plt.yticks(range(len(mc_establish_sent_lex_pos)), [val[0] for val in mc_establish_sent_lex_pos])
    plt.show()
    
    results_establish_zuspitzung = set((makeQuery(establish_zuspitzung)))
    r2 = len(results_establish_zuspitzung)
    print('Anzahl gefundener S‰tze f¸r Establish-Zuspitzung: %s ' % r2)
    mc_establish_zuspitzung = filterResults(tokenizeResults(results_establish_zuspitzung)).most_common(20)
    plt.barh(range(len(mc_establish_zuspitzung)),[val[1] for val in mc_establish_zuspitzung], align='center')
    plt.yticks(range(len(mc_establish_zuspitzung)), [val[0] for val in mc_establish_zuspitzung])
    plt.show()
    
    results_establish_konflikt_simple = makeQuery(np.concatenate(establish_konflikt_simple).ravel().tolist())
    r3 = len(results_establish_konflikt_simple)
    print('Anzahl gefundener S‰tze f¸r Establish-Konflikte-Simpel: %s ' % r3)
    mc_establish_konflikt_simple = filterResults(tokenizeResults(results_establish_konflikt_simple)).most_common(20)
    plt.barh(range(len(mc_establish_konflikt_simple)),[val[1] for val in mc_establish_konflikt_simple], align='center')
    plt.yticks(range(len(mc_establish_konflikt_simple)), [val[0] for val in mc_establish_konflikt_simple])
    plt.show()
    
    results_establish_ort = makeQuery(establish_ort)
    r6 = len(results_establish_ort)
    print('Anzahl gefundener S‰tze f¸r Establish-Ort: %s' % r6)
    mc_establish_ort = filterResults(tokenizeResults(results_establish_ort)).most_common(20)
    plt.barh(range(len(mc_establish_ort)),[val[1] for val in mc_establish_ort], align='center')
    plt.yticks(range(len(mc_establish_ort)), [val[0] for val in mc_establish_ort])
    plt.show()

    

def makeBallad():
    
    makeSetting()
    
    lines = [select(similarTokens(random.choice(establish_mensch))),
             select(similarTokens(random.choice(establish_transformation))),
             select(makeQuery((establish_ort))),
             select(similarTokens(random.choice(establish_transformation))),
             
             select(similarTokens(random.choice(establish_mensch))),
             select(similarTokens(random.choice(establish_transformation))),
             select(similarTokens(random.choice(establish_transformation))),
             select(similarTokens(random.choice(establish_transformation))),
             
             select(makeQuery((establish_zuspitzung1))),
             select(makeQuery(np.concatenate(establish_konflikt_simple).ravel().tolist())),
             select(makeQuery(np.concatenate(establish_konflikt_komplex).ravel().tolist())),
             select(makeQuery((establish_zuspitzung2))),
             
             select(similarTokens(random.choice(establish_katharsis))),
             select(similarTokens(random.choice(establish_katharsis))),
             select(similarTokens(random.choice(establish_katharsis))),
             select(similarTokens(random.choice(establish_katharsis)))]
    
    return lines


#print(makeQuery(['bleich']))

#showStats()

#print(formatBallad((sortiere((tempest))),4))

print(formatBallad(formatiere(makeKuckucksEi()), 4))

#print(set(readSentLexPos()))
