"""Contains wrappers for input pre-processing to be done before training and inference"""
import contractions
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer


class PreProcessor:
    """Performs various pre-processing of given input"""

    def __init__(self):
        self.__stemmer = PorterStemmer()
        nltk.download('punkt')
    
    def pre_process(self, reviews):
        """Convert the input text to proper format expected by the model.
        Transforms txt to lowercase, remove contractions/html/puncutations,
        permorm stemming of words

        Args:
            reviews (DataFrame): Input data to be pre-processed

        Returns:
            DataFrame: Pre-processed input
        """
        pre_processed_texts = []
        for review in reviews:
            text = self.__pre_process_item(review)
            pre_processed_texts.append(text)

        return pre_processed_texts

    def __pre_process_item(self, review):
        text = review.lower()
        text = text.apply(lambda x: contractions.fix(x))
        text = self.__remove_html(text)
        text = self.__do_stemming(text)
        text = self.__remove_punctuations(text)
        
        return text    

    def __remove_html(self, text):
        bs_parser = BeautifulSoup(text)
        return bs_parser.getText()

    def __do_stemming(self, text):
        text = ' '.join([self.__stemmer.stem(word) for word in text.split()])
        return text

    def __remove_punctuations(self, text):
        tokens = nltk.word_tokenize(text)
        text = ' '.join([word for word in tokens if word.isalpha()])
        return text

