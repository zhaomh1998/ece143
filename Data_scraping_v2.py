from googletrans import Translator as gt
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from collections import Counter
import articleDateExtractor as ad
import bs4

class string(str):
    def translate(self):
        '''This method translation to the default string objects in python
        input:
        string object in any language
        output:
        translated string object
        '''
        assert isinstance(self,str)
        translator=gt()
        return translator.translate(self,dest='en').text
    
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
        word=[word]
    output=Counter()
    for j in search(Topic, tld="com", num=n, stop=1, pause=2): 
        for k in word:
            print(j)
            print("article_date",ad.extractArticlePublishedDate(j))
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

def webpage_info(url):
    '''This function provides webpage info like server, content type and content length
    input:
    url--> url to be processed
    output:
    webpage info printed on terminal
    '''
    try:
        print("Server: " + resp.headers['server'])
        print("Last modified: " + resp.headers['last-modified'])
        print("Content type: " + resp.headers['content-type'])
        print("Content length: " + resp.headers['content-length'])
    except:
        return "one of the info not present"
    return url

def visible(element):
    '''this function checks if the data is inside body or somewhere else
    input:
    element--> is a beautiful soup element
    '''
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True

def data_from_html(url):
    '''This function returns the text data from html file
    input:
    url--> url of the file to be processed
    outpu:
    data in text format
    '''
    assert isinstance(url,str),"url is not in string format"
    body=requests.get(url, allow_redirects=False)
    soup = BeautifulSoup(body.content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)


def article_year(url):
    '''This function provides the year the article on the given url was published, it returns none if it is not an article
    '''
    assert isinstance(url,str),"url is not in string format
    try:
        return ad.extractArticlePublishedDate(url).year
    except:
        return None
