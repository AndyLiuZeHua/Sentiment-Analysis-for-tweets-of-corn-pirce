# -*- coding: utf-8 -*-

from pycorenlp import StanfordCoreNLP
import pandas as pd
import chardet
nlp = StanfordCoreNLP('http://localhost:9000')
data = pd.read_csv('/home/andy/ML/FL/newtweet.csv')
tweet = data[[4]]

sescore= list()

for i in range(0,len(tweet)):
	text = str(tweet.ix[i])
	if chardet.detect(text)['encoding'] == 'ascii':
		res = nlp.annotate(text,properties={'annotators': 'sentiment','outputFormat': 'json'})
		score = 0
		count = 0
    		for s in res["sentences"]:
    			score = score + int(s["sentimentValue"])
    			count = count + 1
    		score = score/count
    		sescore.append(score)
    	if chardet.detect(text)['encoding'] <> 'ascii':
    		data = data.drop(i)

print(len(sescore))

print(data.shape)

data.insert(7,'sescore',sescore)

print(data)

data.to_csv('/home/andy/ML/FL/tweetSescore.csv')
