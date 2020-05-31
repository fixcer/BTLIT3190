import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_dict():
	folders = "../datasets/"
	data_map = folders + "train"
	tokens = []
	trainer = open(data_map, 'r')
	for line in trainer:
		tmp = ''
		line = line.split()[2:]
		for i in range(0, len(line), 2):
			tmp += line[i] + ' '
		tokens.append(tmp)
		
	trainer.close()

	vectorizer = TfidfVectorizer(min_df=3)
	X = vectorizer.fit_transform(tokens)
	idf = vectorizer.idf_
	tmp = dict(zip(vectorizer.get_feature_names(), idf))
	dicter = sorted(tmp.items(), key=lambda x: x[1], reverse=True)

	return dicter
