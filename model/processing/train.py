"""Contains wrappers for model training"""
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from model.processing.pre import PreProcessor
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

class Train:
    """Wrapper for model training"""

    def __init__(self, df: DataFrame):
        self.__tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.__model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
        self.__pre_processor = PreProcessor()
        df_review = df['review'].apply(lambda x: self.__pre_processor.pre_process(x))
        df_sentiment = df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            df_review, df_sentiment, test_size=0.2
        )

    def train(self):
        """Trains the model using HuggingFace DistilBert transformer.
        80% of the input is used for training and rest is for validation."""
        
        x_train_list = self.x_train.tolist()
        x_train_tok = self.__tokenizer(x_train_list,max_length=100,padding="max_length",truncation=True,return_tensors="tf")
        x_val_list = self.x_val.tolist()
        x_val_tok = self.__tokenizer(x_val_list,max_length=100,padding="max_length",truncation=True,return_tensors="tf")

        train_ds = tf.data.Dataset.from_tensor_slices((x_train_tok.data,self.y_train)).batch(16)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val_tok.data,self.y_val)).batch(16)
        
        self.__model.compile()
        self.__model.fit(train_ds,validation_data=val_ds)
