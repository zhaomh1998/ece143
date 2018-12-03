from googletrans import Translator as gt
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from collections import Counter

class string(str):
    def translate(self):
        '''This method adds to the translation to the default string objects in python
        input:
        string object in any language
        output:
        translated string object
        '''
        assert isinstance(self,str)
        translator=gt()
        return translator.translate(self,dest=en).text
def web_search(Topic,word,n):
    '''This function searches google to check the web for word frequencies in most ranked articles 
    input:
    Topic (string object)--> the topic to be searched 
    word (list object)--> the word to be identified for that topic 
    n(int)--> the number of webpages to be searched
    output--> word_counts
    '''
    assert isinstance(Topic,str),"input topic is not a string"
    assert (isinstance(word,list) & all(isinstance(j,str) for j in word)) | isinstance(word, str) ,"input word is not a string"
    if isinstance(word,str):
        word=list(word)
    output=Counter()
    for j in search(Topic, tld="com", num=n, stop=1, pause=2): 
        for k in word:
            output[k]+=count_words(j,k)
    return output


    
def count_words(url, word):
    '''This function counts the number of times a word i read on a webpage
    input:
    url--> url of the webpage to be searched
    word--> the word to be counted
    '''
    r = requests.get(url, allow_redirects=False)
    data = BeautifulSoup(r.content, 'html5lib')
    words = data.find(text=lambda text: text and word in text)
    if words==None:
        return 0
    return len(words)
