import os
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

STOPWORDS = set(stopwords.words('english'))

folders = "../datasets"
data_map = folders + "/emails.csv"

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN


def preprocess(sentence):
	sentence = sentence.lower()
	stemmer = SnowballStemmer("english")
	lemmatizer = WordNetLemmatizer()
	tokenizer = RegexpTokenizer(r'[a-z]{3,}')
	tokens = tokenizer.tokenize(sentence)
	tokens = [token for token in tokens if token not in STOPWORDS]
	# tokens = [stemmer.stem(token) for token in tokens]
	tokens_pos = pos_tag(tokens)
	tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in tokens_pos]

	word_freq = Counter(tokens)
	tmp = ''
	for word, freq in word_freq.items():
		tmp += ' ' + word + ' ' + str(freq)
	return tmp


def draw(): 
	plt.figure(figsize=(20,6))

	plt.subplot(1,3,1)
	plt.pie(y_train.value_counts(), colors=['yellowgreen', 'gold'], autopct='%1.1f%%', startangle=90)
	plt.title('Train Set')
	plt.axis('square')

	plt.subplot(1,3,2)
	plt.pie(y_val.value_counts(), colors=['yellowgreen', 'gold'], autopct='%1.1f%%', startangle=90)
	plt.title('Validation Set')
	plt.axis('square')

	plt.subplot(1,3,3)
	plt.pie(y_test.value_counts(), colors=['yellowgreen', 'gold'], autopct='%1.1f%%', startangle=90)
	plt.title('Test Set')
	plt.axis('square')
	plt.legend(['Ham','Spam'], loc=1)

	plt.show()


def write_file(datasets, labels, gen_id):
	dir_file = folders + gen_id
	with open(dir_file, 'w') as f:
		for data, label in zip(datasets, labels):
			f.write(gen_id + ' ' + label + data + '\n')


df = pd.read_csv(data_map)
df = df[pd.notnull(df['labels'])]

df['text'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.labels, test_size=.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


write_file(X_train, y_train, "/train")
write_file(X_val, y_val, "/val")
write_file(X_test, y_test, "/test")

draw()
