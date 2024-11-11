print('this is da FUNC')

'''#!pip uninstall newsapi
!pip install newsapi-python
!pip install transformers
!pip install tensorflow
!pip3 install torch torchvision torchaudio
!pip install numpy
!pip install scipy
!pip install textBlob
!pip install newspaper3k
!pip install SentencePiece
!pip install spacy
!python -m spacy download en_core_web_md '''


'''@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}'''


from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import requests
import re
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
regex = r'([A-z][^.!?]*[.!?]*"?)' # sentences = rc.findall(regex, text) to get sentences

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from newsapi import NewsApiClient
import pandas as pd
import tensorflow as tf
import torch
from transformers import pipeline
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from textblob import TextBlob
from newspaper import Article

class Document:
    link = ''
    summary = ''
    text = ''
    def __init__(self, l, s, t):
      self.link = l
      self.summary = s
      self.text = t

    def get_link(self):
      return self.link

    def get_text(self):
      return self.text

    def get_summary(self):
      return self.summary

### QUESTION_GENERATION MODEL
model_name = "allenai/t5-small-squad2-question-generation"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
def run_QG_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output

### QUESTION_ANSWERING MODEL
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
def run_QA_model(question, context):

    result = question_answerer(question=question, context=context)
    print(result['answer'])
    return result['answer'], result['score']
  

apiKey = 'af1752b5cbfe44fcb3e41e452ca10881'
newsApi = NewsApiClient(api_key=apiKey)
### GENERAL TEXT PARSER (STANDARD)
def general_text_parser(url):
    print(url)
    r = Request(url, headers=headers)
    try:
      html = urlopen(r).read()
      soup = BeautifulSoup(html, features="html.parser")

      # kill all script and style elements
      for script in soup(["script", "style"]):
          script.extract()    # rip it out

      # get text
      text = soup.get_text()

      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in text.splitlines())
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      # drop blank lines
      text = '\n'.join(chunk for chunk in chunks if chunk)

      return text
    
    except Exception as e:
      return "NA"

### GENERAL TEXT PARSER (NEWSPAPER3K)
def general_text_parser_newspaper3k(url):
    print(url)
    
    try:
      article = Article(url)
      article.download()
      article.parse()
      text = article.text
      return text
    
    except Exception as e:
      return "NA"

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
def text_summarizer(text):
    rv = summarizer(text, min_length=5, max_length=100)
    print(rv)
    if (len(rv) !=0):
        return rv[0]["summary_text"]
    return "NA"


### WESBSITE SCRAPER (implements general_text_parser)
def newsApi_text_scraper(topic):
    topic_articles = newsApi.get_everything(q=topic)['articles']
    topic_urls = [x['url'] for x in topic_articles]
    topic_texts = [general_text_parser(url) for url in topic_urls]
    
    topic_docs = [Document(topic_urls[i], text_summarizer(topic_texts[i])) for i in range(len(topic_texts))]

    return topic_docs

### WESBSITE SCRAPER (implements general_text_parser_newspaper3k)
def newsApi_text_scraper_newspaper3k(topic):
    topic_articles = newsApi.get_everything(q=topic)['articles']
    topic_urls = [x['url'] for x in topic_articles]
    topic_texts = [general_text_parser_newspaper3k(url) for url in topic_urls]
    
    topic_docs = [Document(topic_urls[i], text_summarizer(topic_texts[i])) for i in range(len(topic_texts))]

    return topic_docs

#newsApi_text_scraper_newspaper3k("NASDAQ")[10].text

#***RERUN THIS BLOCK WHENEVER U WANNA REINITIALIZE THE SET_OF_ALL_DOCUMENTS SET****
list_of_topics = ["NASDAQ", "S&P 500", "economics", "politics", "America", "Democrat", "Republican", "China"]
set_of_all_documents = set()
question_document_dict = {}

import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load('en_core_web_md')

### Create and maintain set_of_all_documents
def retrieve_documents(topic):
    documents = newsApi_text_scraper_newspaper3k(topic)
    set_of_all_documents.update(documents)

### Retrieve all relevant documents given a question, is SLOW SEARCH
def get_all_relevant_documents(question):
    docs_with_answers = set()
    ###TODO: add vector operations to find k-nearest questestions, add to docs_with_answers

    for document in set_of_all_documents:
      if (len(document.text) != 0):
        answer, score = run_QA_model(question, document.text)
        print(score)
        if (score >= 0.6):
            docs_with_answers.add(document)


    return docs_with_answers

### creates question_document_dict
def create_question_document_dict():
    for document in set_of_all_documents:
        try:
            sentences = sent_tokenize(document.text)
            for sentence in sentences:
                question = run_QA_model(sentence)
                documents = get_all_relevant_documents(question[0])
            ### DOES NOT INCLUDE K_NEAREST_DOCUMENTS: documents.update(k_nearest_documents(question))
                question_document_dict[question] = documents
        except Exception as e:
            pass
### maintains question_document_dict
def maintain_question_document_dict(question, documents):
    question_document_dict[question] = documents


def vectorize(sentence):
    doc = nlp(sentence)
    return doc.vector


def k_nearest_documents(question):
    questions = list(question_document_dict) ## should return all keys as a subscriptable list
    questions.insert(0, question) ## adds given question to the beginning of the list

    vectors = [vectorize(q) for q in questions]

    similarity_matrix = cosine_similarity(vectors)    
    k = 4  # number of most similar sentences to retrieve

    similarities = np.argsort(-similarity_matrix)[:, 1:k+1]
    documents = []
    for i in range(similarities[0]):
        documents.extend(question_document_dict[questions[i]])


    ## update question_document_dict

    return documents

def search_engine_NLP(question):
    if len(question_document_dict) == 0:
        create_question_document_dict()

    if question in list(question_document_dict):
        docs = question_document_dict[question]
        docs.update(k_nearest_documents(question))
        return docs
    
    ## Add (threading I think??) to allow quick pullup of k_nearest_documents and slow pullup of more-direct from set_of_all_documents

    docs = get_all_relevant_documents(question)
    docs.update(k_nearest_documents(question))
    maintain_question_document_dict(question, docs)
    
    return docs

    
        
