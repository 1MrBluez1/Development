{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#Testing and Training NLP for Fake News Detection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import nltk\n",
    "import re\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize, sent_tokenize\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing import text, sequence\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, GRU\n",
    "from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping\n",
    "from tensorflow.keras.callbacks import History \n",
    "\n",
    "from wordcloud import WordCloud, STOPWORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  \n",
       "0  December 31, 2017   \n",
       "1  December 29, 2017   \n",
       "2  December 31, 2017   \n",
       "3  December 30, 2017   \n",
       "4  December 29, 2017   "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "real = pd.read_csv('Fake-News\\True.csv')\n",
    "real.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "\n",
       "                date  \n",
       "0  December 31, 2017  \n",
       "1  December 31, 2017  \n",
       "2  December 30, 2017  \n",
       "3  December 29, 2017  \n",
       "4  December 25, 2017  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fake = pd.read_csv('Fake-News\\Fake.csv')\n",
    "fake.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "real['Category'] = 1\n",
    "fake['Category'] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(21417, 5)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>Category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  Category  \n",
       "0  December 31, 2017          1  \n",
       "1  December 29, 2017          1  \n",
       "2  December 31, 2017          1  \n",
       "3  December 30, 2017          1  \n",
       "4  December 29, 2017          1  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(real.shape)\n",
    "real.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(23481, 5)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>Category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "\n",
       "                date  Category  \n",
       "0  December 31, 2017         0  \n",
       "1  December 31, 2017         0  \n",
       "2  December 30, 2017         0  \n",
       "3  December 29, 2017         0  \n",
       "4  December 25, 2017         0  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(fake.shape)\n",
    "fake.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.concat([real, fake]).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(44898, 5)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>Category</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  Category  \n",
       "0  December 31, 2017          1  \n",
       "1  December 29, 2017          1  \n",
       "2  December 31, 2017          1  \n",
       "3  December 30, 2017          1  \n",
       "4  December 29, 2017          1  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(dataset.shape)\n",
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "80"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import gc\n",
    "del [[real,fake]]\n",
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title       0\n",
       "text        0\n",
       "subject     0\n",
       "date        0\n",
       "Category    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    As U.S. budget fight looms, Republicans flip t...\n",
       "1    U.S. military to accept transgender recruits o...\n",
       "2    Senior U.S. Republican senator: 'Let Mr. Muell...\n",
       "3    FBI Russia probe helped by Australian diplomat...\n",
       "4    Trump wants Postal Service to charge 'much mor...\n",
       "Name: final_text, dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset['final_text'] = dataset['title'] + dataset['text']\n",
    "dataset['final_text'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    23481\n",
       "1    21417\n",
       "Name: Category, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset['Category'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x17e4330aa48>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(dataset[\"Category\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th>final_text</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Category</th>\n",
       "      <th>subject</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th rowspan=\"6\" valign=\"top\">0</th>\n",
       "      <th>Government News</th>\n",
       "      <td>1570</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Middle-east</th>\n",
       "      <td>778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>News</th>\n",
       "      <td>9050</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>US_News</th>\n",
       "      <td>783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>left-news</th>\n",
       "      <td>4459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>politics</th>\n",
       "      <td>6841</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th rowspan=\"2\" valign=\"top\">1</th>\n",
       "      <th>politicsNews</th>\n",
       "      <td>11272</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>worldnews</th>\n",
       "      <td>10145</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                          final_text\n",
       "Category subject                    \n",
       "0        Government News        1570\n",
       "         Middle-east             778\n",
       "         News                   9050\n",
       "         US_News                 783\n",
       "         left-news              4459\n",
       "         politics               6841\n",
       "1        politicsNews          11272\n",
       "         worldnews             10145"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset[['Category','subject','final_text']].groupby(['Category','subject']).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x17e4961ff08>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAnAAAAE+CAYAAAANqS0iAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3deZwdZZ3v8c8PCIZ9DY7QYAdBEZAtAXFQYEDZvCagoGR0kgBeRBF07siIjFe2wYkGRQSXi8oqixgGg4ASRgmgw5YAAyEMhJ0GhLCTwQAJv/tHPR0OzeklobtPV/J5v1796qqnnqp6qvpUn+956lRVZCaSJEmqj+Va3QBJkiQtHgOcJElSzRjgJEmSasYAJ0mSVDMGOEmSpJoxwEmSJNXMCq1uwGBbd911s729vdXNkCRJ6tXMmTOfzswRXcuXuQDX3t7OjBkzWt0MSZKkXkXEw83KPYUqSZJUMwY4SZKkmjHASZIk1cwy9x04SZJUb6+99hodHR3Mnz+/1U3pN8OHD6etrY1hw4b1qb4BTpIk1UpHRwerrbYa7e3tRESrm/O2ZSbPPPMMHR0djBw5sk/zeApVkiTVyvz581lnnXWWivAGEBGss846i9WjaICTJEm1s7SEt06Luz0GOEmStFT4y1/+woEHHsh73vMeNt98c/bZZx/uvffepnWff/55fvzjHw9yC/uPAU6SJNVeZrLffvux6667cv/99zN79my+/e1v8+STTzatP1gBbsGCBQOyXAOcJEmqvWuuuYZhw4Zx2GGHLSrbZptt2Hbbbdl9993Zbrvt+MAHPsDUqVMBOProo7n//vvZZpttOOqoowCYPHky22+/PVtttRXHHnvsouWceOKJbLbZZnzsYx9j3LhxnHzyyQDcfvvt7Ljjjmy11Vbst99+PPfccwDsuuuuHHPMMeyyyy6cdNJJjBw5ktdeew2AF198kfb29kXjS8qrUCVJUu3NmjWLUaNGvaV8+PDhXHrppay++uo8/fTT7LjjjowZM4ZJkyYxa9Ysbr/9dgCmTZvGnDlzuPnmm8lMxowZw3XXXcfKK6/MJZdcwm233caCBQvYbrvtFq1n/PjxnHbaaeyyyy5861vf4vjjj+cHP/gBUPXwXXvttQA89NBDXHHFFey7775cdNFFfOpTn+rz7UK6Y4BrkUdO+MCgrm+jb905qOuTJGkoyEyOOeYYrrvuOpZbbjkee+yxpqdVp02bxrRp09h2220BmDdvHnPmzOGll15i7NixrLTSSgB84hOfAOCFF17g+eefZ5dddgFgwoQJHHDAAYuW95nPfGbR8Oc//3m++93vsu+++3LWWWfxs5/97G1vlwFOkiTV3hZbbMGUKVPeUn7++eczd+5cZs6cybBhw2hvb296u47M5Bvf+AZf+MIX3lR+yimnLFF7VllllUXDO+20Ew899BDXXnstCxcuZMstt1yiZTbyO3CSJKn2dtttN1555ZU39W7dcsstPPzww6y33noMGzaMa665hocffhiA1VZbjZdeemlR3T333JMzzzyTefPmAfDYY4/x1FNP8eEPf5jf/va3zJ8/n3nz5nHFFVcAsMYaa7DWWmtx/fXXA3Deeect6o1rZvz48YwbN46DDjqoX7bXHjhJklR7EcGll17KV7/6VSZNmsTw4cNpb2/nuOOO48gjj2T06NFss802bLbZZgCss8467LTTTmy55ZbsvffeTJ48mbvvvpsPfehDAKy66qr88pe/ZPvtt2fMmDFsvfXWvPvd72b06NGsscYaAJxzzjkcdthhvPzyy2y88cacddZZ3bbvs5/9LN/85jcZN25c/2xvZvbLgupi9OjROWPGjFY3w+/ASZK0hO6++27e//73D9r65s2bx6qrrsrLL7/MzjvvzBlnnMF22223WMuYMmUKU6dO5bzzzuu2TrPtioiZmTm6a1174CRJknpw6KGHMnv2bObPn8+ECRMWO7wdccQR/O53v+PKK6/stzYZ4CRJknpwwQUXvK35TzvttH5qyRu8iEGSJKlmDHCSJEk1Y4CTJEmqGQOcJElSzRjgJEmS+snvf/973ve+97HJJpswadKkAVuPV6FKkqSl0qijzu3X5c2cPL7H6QsXLuTwww/n6quvpq2tbdFNgDfffPN+bQfYAydJktQvbr75ZjbZZBM23nhjVlxxRQ488ECmTp06IOsywEmSJPWDxx57jA033HDReFtbG4899tiArMsAJ0mS1A+aPZ40IgZkXQY4SZKkftDW1sajjz66aLyjo4P1119/QNZlgJMkSeoH22+/PXPmzOHBBx/k1Vdf5aKLLmLMmDEDsi6vQpUkSeoHK6ywAqeffjp77rknCxcu5OCDD2aLLbYYmHUNyFIlSZJarLfb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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.countplot(x= \"subject\", hue = \"Category\", data=dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_text_result = []\n",
    "for text in dataset['final_text']:\n",
    "    result = re.sub('[^a-zA-Z]', ' ', text)\n",
    "    result = result.lower()\n",
    "    result = result.split()\n",
    "    result = [r for r in result if r not in set(stopwords.words('english'))]\n",
    "    final_text_result.append(\" \".join(result))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "44898\n"
     ]
    }
   ],
   "source": [
    "print(len(final_text_result))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_final = np.array(final_text_result)\n",
    "y_final = dataset['Category']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.3, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing import text\n",
    "max_words = 10000\n",
    "tokenizer = text.Tokenizer(num_words=max_words)\n",
    "tokenizer.fit_on_texts(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate sequence of Tokens\n",
    "X_train_sequence = tokenizer.texts_to_sequences(X_train)\n",
    "X_test_sequence = tokenizer.texts_to_sequences(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pad the sequences\n",
    "sent_length = 400\n",
    "X_train_pad = sequence.pad_sequences(X_train_sequence, maxlen=sent_length)\n",
    "X_test_pad = sequence.pad_sequences(X_test_sequence, maxlen=sent_length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "word_index = tokenizer.word_index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# reduce memory\n",
    "def reduce_mem_usage(df, verbose=True):\n",
    "    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n",
    "    start_mem = df.memory_usage().sum() / 1024**2\n",
    "    for col in df.columns:\n",
    "        col_type = df[col].dtypes\n",
    "        if col_type in numerics:\n",
    "            c_min = df[col].min()\n",
    "            c_max = df[col].max()\n",
    "            if str(col_type)[:3] == 'int':\n",
    "                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n",
    "                    df[col] = df[col].astype(np.int8)\n",
    "                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n",
    "                    df[col] = df[col].astype(np.int16)\n",
    "                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n",
    "                    df[col] = df[col].astype(np.int32)\n",
    "                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n",
    "                    df[col] = df[col].astype(np.int64)\n",
    "            else:\n",
    "                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n",
    "                    df[col] = df[col].astype(np.float16)\n",
    "                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n",
    "                    df[col] = df[col].astype(np.float32)\n",
    "                else:\n",
    "                    df[col] = df[col].astype(np.float64)\n",
    "\n",
    "    end_mem = df.memory_usage().sum() / 1024**2\n",
    "    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))\n",
    "    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))\n",
    "\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Memory usage after optimization is: 1.76 MB\n",
      "Decreased by 14.6%\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "20"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = reduce_mem_usage(dataset)\n",
    "del final_text_result\n",
    "del X_final\n",
    "del y_final\n",
    "del X_train\n",
    "del X_test\n",
    "del X_train_sequence\n",
    "del X_test_sequence\n",
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "GLOVE_EMBEDDINGS_FILE = 'C:\\\\Users\\\\Gandalf\\Fake-News\\glove.twitter.27B.100d.txt'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size of vocabulary in GloVe:  1193514\n",
      "Wall time: 40.3 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "embedding_vectors = {}\n",
    "with open(GLOVE_EMBEDDINGS_FILE,'r',encoding='utf-8') as file:\n",
    "    for row in file:\n",
    "        values = row.split(' ')\n",
    "        word = values[0]\n",
    "        weights = np.asarray([float(val) for val in values[1:]])\n",
    "        embedding_vectors[word] = weights\n",
    "print(\"Size of vocabulary in GloVe: \", len(embedding_vectors))  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "emb_dim = 100\n",
    "if max_words is not None: \n",
    "    vocab_len = max_words \n",
    "else:\n",
    "    vocab_len = len(word_index)+1\n",
    "embedding_matrix = np.zeros((vocab_len, emb_dim))\n",
    "oov_count = 0\n",
    "oov_words = []\n",
    "for word, idx in word_index.items():\n",
    "    if idx < vocab_len:\n",
    "        embedding_vector = embedding_vectors.get(word)\n",
    "        if embedding_vector is not None:\n",
    "            embedding_matrix[idx] = embedding_vector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)\n",
    "early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding (Embedding)        (None, None, 100)         1000000   \n",
      "_________________________________________________________________\n",
      "lstm (LSTM)                  (None, 128)               117248    \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 1)                 129       \n",
      "=================================================================\n",
      "Total params: 1,117,377\n",
      "Trainable params: 117,377\n",
      "Non-trainable params: 1,000,000\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "lstm_model = Sequential()\n",
    "lstm_model.add(Embedding(vocab_len, emb_dim, trainable = False, weights=[embedding_matrix]))\n",
    "lstm_model.add(LSTM(128, return_sequences=False))\n",
    "# lstm_model.add(Dropout(0.25))\n",
    "lstm_model.add(Dense(1, activation = 'sigmoid'))\n",
    "lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "print(lstm_model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "123/123 [==============================] - 179s 1s/step - loss: 0.2128 - accuracy: 0.9186 - val_loss: 0.1146 - val_accuracy: 0.9602 - lr: 0.0010\n",
      "Epoch 2/20\n",
      "123/123 [==============================] - 181s 1s/step - loss: 0.1384 - accuracy: 0.9492 - val_loss: 0.5353 - val_accuracy: 0.7861 - lr: 0.0010\n",
      "Epoch 3/20\n",
      "123/123 [==============================] - 181s 1s/step - loss: 0.3619 - accuracy: 0.8549 - val_loss: 0.2697 - val_accuracy: 0.8901 - lr: 0.0010\n",
      "Epoch 4/20\n",
      "123/123 [==============================] - 180s 1s/step - loss: 0.2422 - accuracy: 0.9014 - val_loss: 0.2198 - val_accuracy: 0.9099 - lr: 0.0010\n",
      "Epoch 5/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1918 - accuracy: 0.9253 - val_loss: 0.1664 - val_accuracy: 0.9350 - lr: 0.0010\n",
      "Epoch 6/20\n",
      "123/123 [==============================] - ETA: 0s - loss: 0.1409 - accuracy: 0.9492\n",
      "Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.\n",
      "123/123 [==============================] - 177s 1s/step - loss: 0.1409 - accuracy: 0.9492 - val_loss: 0.1216 - val_accuracy: 0.9597 - lr: 0.0010\n",
      "Epoch 7/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1298 - accuracy: 0.9559 - val_loss: 0.1146 - val_accuracy: 0.9630 - lr: 1.0000e-04\n",
      "Epoch 8/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1212 - accuracy: 0.9597 - val_loss: 0.1147 - val_accuracy: 0.9637 - lr: 1.0000e-04\n",
      "Epoch 9/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1160 - accuracy: 0.9619 - val_loss: 0.1144 - val_accuracy: 0.9630 - lr: 1.0000e-04\n",
      "Epoch 10/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1170 - accuracy: 0.9607 - val_loss: 0.1131 - val_accuracy: 0.9633 - lr: 1.0000e-04\n",
      "Epoch 11/20\n",
      "123/123 [==============================] - 179s 1s/step - loss: 0.1148 - accuracy: 0.9614 - val_loss: 0.1125 - val_accuracy: 0.9633 - lr: 1.0000e-04\n",
      "Epoch 12/20\n",
      "123/123 [==============================] - 179s 1s/step - loss: 0.1154 - accuracy: 0.9611 - val_loss: 0.1216 - val_accuracy: 0.9597 - lr: 1.0000e-04\n",
      "Epoch 13/20\n",
      "123/123 [==============================] - 179s 1s/step - loss: 0.1194 - accuracy: 0.9601 - val_loss: 0.1100 - val_accuracy: 0.9659 - lr: 1.0000e-04\n",
      "Epoch 14/20\n",
      "123/123 [==============================] - 177s 1s/step - loss: 0.1123 - accuracy: 0.9632 - val_loss: 0.1118 - val_accuracy: 0.9635 - lr: 1.0000e-04\n",
      "Epoch 15/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1920 - accuracy: 0.9312 - val_loss: 0.3642 - val_accuracy: 0.8616 - lr: 1.0000e-04\n",
      "Epoch 16/20\n",
      "123/123 [==============================] - 177s 1s/step - loss: 0.2305 - accuracy: 0.9129 - val_loss: 0.1560 - val_accuracy: 0.9477 - lr: 1.0000e-04\n",
      "Epoch 17/20\n",
      "123/123 [==============================] - 178s 1s/step - loss: 0.1412 - accuracy: 0.9552 - val_loss: 0.1317 - val_accuracy: 0.9598 - lr: 1.0000e-04\n",
      "Epoch 18/20\n",
      "123/123 [==============================] - ETA: 0s - loss: 0.1250 - accuracy: 0.9612\n",
      "Epoch 00018: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.\n",
      "123/123 [==============================] - 176s 1s/step - loss: 0.1250 - accuracy: 0.9612 - val_loss: 0.1214 - val_accuracy: 0.9633 - lr: 1.0000e-04\n",
      "Epoch 19/20\n",
      "123/123 [==============================] - 176s 1s/step - loss: 0.1203 - accuracy: 0.9627 - val_loss: 0.1205 - val_accuracy: 0.9634 - lr: 1.0000e-05\n",
      "Epoch 20/20\n",
      "123/123 [==============================] - 174s 1s/step - loss: 0.1196 - accuracy: 0.9627 - val_loss: 0.1195 - val_accuracy: 0.9635 - lr: 1.0000e-05\n",
      "Wall time: 59min 47s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "seq_model1 = lstm_model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=20, batch_size = 256, callbacks=([reduce_lr, early_stop]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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iJQSY0xiebdB8tMxmIiISNVW0DGYhtSJiIiIiIiIiIhXKXASERERERERERGvUuAkIiIiIiIiIiJedV7M4eRORkYGu3btIj09vbybImUkICCAZs2a4e/vX95NEREREREREQ/UX694itOfPm8Dp127dhEcHEx4eDjGmPJujpQyay2HDh1i165dtGzZsrybIyIiIiIiIh6ov16xFLc/fd4OqUtPT6d+/fr68p4njDHUr19fCbmIiIiIiEgFp/56xVLc/vR5GzgB+vKeZ/R5i4iIiIiIVA7qv1Usxfk8zuvASUREREREREREvE+BUzk5dOgQ0dHRREdHExoaStOmTXPvnzp1qlDbuP3229m0aVOB6/z3v/8lNjbWG00GYN++ffj5+fHmm296bZsiIiIiIiIiFUFl7Kv36NGD1atXe2Vb3nTeThpeVLHrYpm4ZCI7knfQonYLJvWeRExkTLG3V79+/dwvxJNPPknNmjWZMGHCGetYa7HW4uPjPhd8++23z7mfe+65p9htdOe9996jW7duxMXFMWrUKK9uO6/MzEz8/PT1FBERERERkYJ5s79eWfvqFZEqnAohdl0sY+aPITE5EYslMTmRMfPHELvOe5VDObZs2UJERARjx46lU6dO7NmzhzFjxtC5c2fat2/P008/nbtuToqZmZlJnTp1ePTRR4mKiqJbt27s378fgL/+9a9Mnjw5d/1HH32Url270qZNG3766ScAUlNTGThwIFFRUQwdOpTOnTt7TEfj4uKYPHky27ZtY+/evbnLP//8czp16kRUVBR9+/YF4Pjx44wcOZLIyEg6dOjAxx9/nNvWHHPmzGH06NEA3HrrrTz00EP06tWLxx9/nGXLltGtWzc6duxI9+7d2bx5M+CEUePHjyciIoIOHTowdepUFi1axKBBg3K3u2DBAm655ZYSfx4iIiIiIiJScZVVf72i99XPduLEidz+eKdOnfjuu+8AWLduHV26dCE6OpoOHTqwbds2jh8/zjXXXENUVBQRERF88MEHXnnPVEICPLDwAVbv9fyhLdu1jJNZJ89YlpaRxqhPRjF91XS3z4kOjWby1ZOL1Z4NGzbw9ttv89prrwHw/PPPU69ePTIzM+nVqxc333wz7dq1O+M5ycnJXHHFFTz//PM8+OCDvPXWWzz66KP5tm2tZcWKFXz66ac8/fTTLFy4kP/85z+Ehoby4YcfsmbNGjp16uS2XQkJCRw5coSLL76Ym2++mblz53L//fezd+9e/vznP/P9998TFhbG4cOHAScNbtiwIevWrcNay9GjR8/52rdu3cqSJUvw8fEhOTmZH374AV9fXxYuXMhf//pX3nvvPV599VV2797NmjVr8PX15fDhw9SpU4f777+fQ4cOUb9+fd5++21uv/32or71IiIiIiIiUoFUpP56Re2ruzNlyhSqVavGunXrWL9+Pddeey2bN29m6tSpTJgwgcGDB3Py5EmstXzyySeEh4ezYMGC3DZ7gyqcCuHsL++5lpdUq1at6NKlS+79uLg4OnXqRKdOndi4cSMbNmzI95zAwECuueYaAC6++GISEhLcbvumm27Kt84PP/zAkCFDAIiKiqJ9+/ZunxsXF8fgwYMBGDJkCHFxcQD873//o1evXoSFhQFQr149AL766qvcMkFjDHXr1j3nax80aFBuWeLRo0e56aabiIiIYMKECaxfvz53u2PHjsXX1zd3fz4+PgwbNozZs2dz+PBhVq1alVtpJSIiIiIiIlVTWfbXK2pf3Z0ffviB4cOHA9C+fXuaNGnCli1buOyyy3j22Wd58cUX2blzJwEBAXTo0IGFCxfy6KOP8uOPP1K7du1C76cgqnCCcyab4ZPDSUxOzLc8rHYYS29b6vX2BAUF5f6+efNmXn75ZVasWEGdOnW49dZbSU9Pz/ecatWq5f7u6+tLZmam221Xr1493zrW2kK1Ky4ujkOHDjFz5kwAdu/ezfbt27HWur1EorvlPj4+Z+zv7NeS97VPnDiRfv36cffdd7Nlyxauvvpqj9sFuOOOOxg4cCAAgwcPzg2kREREREREpHKqSP31itpXd8fTc4cPH063bt34/PPP6dOnDzNnzuTyyy9n5cqVfPHFFzz88MNcd911PP7448Xedw5VOBXCpN6TqOFf44xlNfxrMKn3pFLf97FjxwgODqZWrVrs2bOHRYsWeX0fPXr0YO7cuYAzntNdKrthwwaysrJISkoiISGBhIQEHn74YebMmUP37t35+uuvSUx0/pHnDKnr27cvr7zyCuB82Y8cOYKPjw9169Zl8+bNZGdnM2/ePI/tSk5OpmnTpgDMmDEjd3nfvn159dVXycrKOmN/zZs3p0GDBjz//PPcdtttJXtTREREREREpMIrr/56Remre3L55Zfn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2LW/Gqmz2mBh1AbzVF/X9mmo/eVxMa5/ABEREREREREREZ/QkDoREREREREREfEpBU4iIiIiIiIiIuJTCpxEREREC0iydgAAAoVJREFURERERMSnFDiJiIiIiIiIiIhPKXASERERERERERGfUuAkIkHHGFNljFnittznw3NnGGOW+ep8IiIiIuFA7S8R8bWYQBdARMSDPdba4wNdCBEREZEIovaXiPiUejiJSMgwxuQZY8YbYxa4lh6u7enGmBxjzFLXbZpreydjzFvGmB9dS1/XqaKNMc8ZY5YbY+YYYxJcx99hjFnhOs+rAaqmiIiISNBQ+0tEDpQCJxEJRgn1unRf5rZvp7X2JOA/wETXtv8AL1trjwWygadc258C5lprjwMygeWu7YcDk6y1xwA7gItc2+8DTnCd5y/+qpyIiIhIEFL7S0R8ylhrA10GEZE6jDHF1tpkD9vzgIHW2jXGmFhgi7W2nTHmdyDVWlvh2r7ZWtveGFMAdLHWlrmdIwP4xFp7uOv+vUCstfZfxpiPgGLgbeBta22xn6sqIiIiEhTU/hIRX1MPJxEJNdbLurdjPClzW69i73x25wKTgN7AD8YYzXMnIiIiovaXiBwABU4iEmouc7v91rX+DTDCtZ4FfOVazwFuBjDGRBtjWnk7qTEmCuhqrf0cuAdoDTT4lU9EREQkAqn9JSL7TemxiASjBGPMErf7H1lray7NG2+MmY8TmF/u2nYH8IIx5m9AAXCda/udwFRjzA04v6TdDGz28pzRwP+MMSmAAZ6w1u7wWY1EREREgpvaXyLiU5rDSURChmsOgT7W2t8DXRYRERGRSKD2l4gcKA2pExERERERERERn1IPJxERERERERER8Sn1cBIREREREREREZ9S4CQiIiIiIiIiIj6lwElERERERERERHxKgZOIiIiIiIiIiPiUAicREREREREREfGp/w/drdhhWLi/VgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "epochs = [i for i in range(20)]\n",
    "fig , ax = plt.subplots(1,2)\n",
    "train_acc = seq_model1.history['accuracy']\n",
    "train_loss = seq_model1.history['loss']\n",
    "val_acc = seq_model1.history['val_accuracy']\n",
    "val_loss = seq_model1.history['val_loss']\n",
    "fig.set_size_inches(20,10)\n",
    "\n",
    "ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')\n",
    "ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')\n",
    "ax[0].set_title('Training & Testing Accuracy')\n",
    "ax[0].legend()\n",
    "ax[0].set_xlabel(\"Epochs\")\n",
    "ax[0].set_ylabel(\"Accuracy\")\n",
    "\n",
    "ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')\n",
    "ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')\n",
    "ax[1].set_title('Training & Testing Loss')\n",
    "ax[1].legend()\n",
    "ax[1].set_xlabel(\"Epochs\")\n",
    "ax[1].set_ylabel(\"Loss\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train accuracy: 96.32174968719482\n",
      "Test accuracy: 96.34743928909302\n"
     ]
    }
   ],
   "source": [
    "train_lstm_results = lstm_model.evaluate(X_train_pad, y_train, verbose=0, batch_size=256)\n",
    "test_lstm_results = lstm_model.evaluate(X_test_pad, y_test, verbose=0, batch_size=256)\n",
    "print(\"Train accuracy: {}\".format(train_lstm_results[1]*100))\n",
    "print(\"Test accuracy: {}\".format(test_lstm_results[1]*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-36-d3787c415889>:1: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.\n",
      "Instructions for updating:\n",
      "Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype(\"int32\")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).\n",
      "Accuarcy: 96.35\n"
     ]
    }
   ],
   "source": [
    "y_pred = lstm_model.predict_classes(X_test_pad)\n",
    "print(\"Accuarcy: {}\".format(round(accuracy_score(y_test, y_pred)*100,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      " [[6742  327]\n",
      " [ 165 6236]]\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(\"Confusion Matrix: \\n\", cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report: \n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.95      0.96      7069\n",
      "           1       0.95      0.97      0.96      6401\n",
      "\n",
      "    accuracy                           0.96     13470\n",
      "   macro avg       0.96      0.96      0.96     13470\n",
      "weighted avg       0.96      0.96      0.96     13470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Classification Report: \\n\", classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding_1 (Embedding)      (None, None, 100)         1000000   \n",
      "_________________________________________________________________\n",
      "gru (GRU)                    (None, 128)               88320     \n",
      "_________________________________________________________________\n",
      "dropout (Dropout)            (None, 128)               0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 1)                 129       \n",
      "=================================================================\n",
      "Total params: 1,088,449\n",
      "Trainable params: 88,449\n",
      "Non-trainable params: 1,000,000\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "emb_dim = embedding_matrix.shape[1]\n",
    "gru_model = Sequential()\n",
    "gru_model.add(Embedding(vocab_len, emb_dim, trainable = False, weights=[embedding_matrix]))\n",
    "gru_model.add(GRU(128, return_sequences=False))\n",
    "gru_model.add(Dropout(0.5))\n",
    "gru_model.add(Dense(1, activation = 'sigmoid'))\n",
    "gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "print(gru_model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.2316 - accuracy: 0.9057 - val_loss: 0.1308 - val_accuracy: 0.9549 - lr: 0.0010\n",
      "Epoch 2/20\n",
      "123/123 [==============================] - 159s 1s/step - loss: 0.1227 - accuracy: 0.9584 - val_loss: 0.0927 - val_accuracy: 0.9712 - lr: 0.0010\n",
      "Epoch 3/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0534 - accuracy: 0.9847 - val_loss: 0.0336 - val_accuracy: 0.9904 - lr: 0.0010\n",
      "Epoch 4/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0328 - accuracy: 0.9916 - val_loss: 0.0146 - val_accuracy: 0.9964 - lr: 0.0010\n",
      "Epoch 5/20\n",
      "123/123 [==============================] - 160s 1s/step - loss: 0.0166 - accuracy: 0.9950 - val_loss: 0.0117 - val_accuracy: 0.9966 - lr: 0.0010\n",
      "Epoch 6/20\n",
      "123/123 [==============================] - 159s 1s/step - loss: 0.0114 - accuracy: 0.9966 - val_loss: 0.0091 - val_accuracy: 0.9973 - lr: 0.0010\n",
      "Epoch 7/20\n",
      "123/123 [==============================] - 159s 1s/step - loss: 0.0091 - accuracy: 0.9974 - val_loss: 0.0080 - val_accuracy: 0.9977 - lr: 0.0010\n",
      "Epoch 8/20\n",
      "123/123 [==============================] - 160s 1s/step - loss: 0.0076 - accuracy: 0.9978 - val_loss: 0.0087 - val_accuracy: 0.9976 - lr: 0.0010\n",
      "Epoch 9/20\n",
      "123/123 [==============================] - 159s 1s/step - loss: 0.0062 - accuracy: 0.9984 - val_loss: 0.0080 - val_accuracy: 0.9980 - lr: 0.0010\n",
      "Epoch 10/20\n",
      "123/123 [==============================] - 159s 1s/step - loss: 0.0059 - accuracy: 0.9982 - val_loss: 0.0072 - val_accuracy: 0.9982 - lr: 0.0010\n",
      "Epoch 11/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0056 - accuracy: 0.9981 - val_loss: 0.0072 - val_accuracy: 0.9980 - lr: 0.0010\n",
      "Epoch 12/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0042 - accuracy: 0.9989 - val_loss: 0.0071 - val_accuracy: 0.9981 - lr: 0.0010\n",
      "Epoch 13/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0031 - accuracy: 0.9991 - val_loss: 0.0112 - val_accuracy: 0.9970 - lr: 0.0010\n",
      "Epoch 14/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0031 - accuracy: 0.9991 - val_loss: 0.0078 - val_accuracy: 0.9977 - lr: 0.0010\n",
      "Epoch 15/20\n",
      "123/123 [==============================] - ETA: 0s - loss: 0.0025 - accuracy: 0.9990\n",
      "Epoch 00015: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0025 - accuracy: 0.9990 - val_loss: 0.0075 - val_accuracy: 0.9976 - lr: 0.0010\n",
      "Epoch 16/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0013 - accuracy: 0.9997 - val_loss: 0.0065 - val_accuracy: 0.9984 - lr: 1.0000e-04\n",
      "Epoch 17/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0011 - accuracy: 0.9998 - val_loss: 0.0062 - val_accuracy: 0.9987 - lr: 1.0000e-04\n",
      "Epoch 18/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.0063 - val_accuracy: 0.9985 - lr: 1.0000e-04\n",
      "Epoch 19/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.0065 - val_accuracy: 0.9984 - lr: 1.0000e-04\n",
      "Epoch 20/20\n",
      "123/123 [==============================] - 161s 1s/step - loss: 0.0011 - accuracy: 0.9997 - val_loss: 0.0059 - val_accuracy: 0.9987 - lr: 1.0000e-04\n",
      "Wall time: 53min 55s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "seq_model2 = gru_model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=20, batch_size = 256, callbacks=([reduce_lr, early_stop]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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dBgiIiLiR8aYjEDHIOXpGExERKRxq+r4S9PfRERERERERESk2pRUEhERERERERGRalNSSUREREREREREqq3R9FTypbCwkG3btpGXlxfoUKQOhYeHk5iYSGhoaKBDERERERERER90vl7/1ORculEnlbZt24bL5aJjx44YYwIdjtQBay2ZmZls27aNTp06BTocERERERER8UHn6/VLTc+lG/X0t7y8PFq1aqUvaBNijKFVq1bKdouIiIiIiNRjOl+vX2p6Lt2ok0qAvqBNkP7mIiIiIiIi9Z/O3eqXmvw9Gn1SSUREREREREREap+SSn6UmZlJr1696NWrF/Hx8bjd7tLnBQUFxzXGtddey48//ljlNs899xxpaWm1ETIAv/zyCyEhIfzjH/+otTFFRERERERE6ouGeL4+YMAAVqxYUStj1ZZG3ai7utLS05i8YDJbsrbQoUUHpg6ayuiU0TUer1WrVqV/8AceeICoqCgmTpxYbhtrLdZagoJ85/deffXVY+7n5ptvrnGMvrz11lv069ePWbNmMW7cuFoduyyPx0NIiL6CIiIiIiIiUjWdr9dPqlTySktPI3VeKhlZGVgsGVkZpM5LJS299iqASmzYsIHk5GQmTJhA79692blzJ6mpqfTp04fu3bvz0EMPlW5bkon0eDzExMQwadIkevbsSb9+/di9ezcAf/nLX3jqqadKt580aRJ9+/bltNNO46uvvgIgJyeHYcOG0bNnT0aNGkWfPn0qzXDOmjWLp556io0bN7Jr167S5R999BG9e/emZ8+eDB48GIDs7GyuueYaUlJS6NGjB3PmzCmNtcTs2bO5/vrrARgzZgx33XUX55xzDvfeey/ffPMN/fr14/TTT6d///6sX78ecBJOf/zjH0lOTqZHjx7MmDGDTz/9lOHDh5eOO3/+fK644ooT/nuIiIiIiIhI/aXz9codPny49Jy8d+/eLFy4EID09HR+9atf0atXL3r06MHGjRvJzs5myJAh9OzZk+TkZN59990T/ryaTJnIHZ/cwYpdlf9Rvtn2DflF+eWW5RbmMm7uOF767iWfr+kV34unLniqRvH88MMPvPrqq7zwwgsATJs2jZYtW+LxeDjnnHO4/PLL6datW7nXZGVlcfbZZzNt2jTuvPNOXnnlFSZNmlRhbGsty5Yt48MPP+Shhx7ik08+4dlnnyU+Pp733nuPlStX0rt3b59xbd68mf3793PGGWdw+eWX8/bbb3Pbbbexa9cubrzxRhYtWkRSUhL79u0DnIxu69atSU9Px1rLgQMHjvnef/75ZxYsWEBQUBBZWVksXryY4OBgPvnkE/7yl7/w1ltv8fzzz7Njxw5WrlxJcHAw+/btIyYmhttuu43MzExatWrFq6++yrXXXlvdj15ERERERETqEZ2vH9/5ui/PPPMMYWFhpKens2bNGi688ELWr1/PjBkzmDhxIiNGjCA/Px9rLXPnzqVjx47Mnz+/NOYTpUolr6O/oMdafqJOPvlkfvWrX5U+nzVrFr1796Z3796sXbuWH374ocJrIiIiGDJkCABnnHEGmzdv9jn2ZZddVmGbxYsXM3LkSAB69uxJ9+7dfb521qxZjBgxAoCRI0cya9YsAL7++mvOOecckpKSAGjZsiUAn332WWk5nzGG2NjYY7734cOHl5YPHjhwgMsuu4zk5GQmTpzImjVrSsedMGECwcHBpfsLCgriyiuv5M0332Tfvn189913pRVTIiIiIiIi0jjpfL1yixcv5qqrrgKge/fuJCQksGHDBs466yymTJnC448/ztatWwkPD6dHjx588sknTJo0iSVLltCiRYvj3k9lmkyl0rEylB2f6khGVkaF5Uktkvhi7Be1Hk/z5s1LH69fv56nn36aZcuWERMTw5gxY8jLy6vwmrCwsNLHwcHBeDwen2M3a9aswjbW2uOKa9asWWRmZvLaa68BsGPHDjZt2oS11uflBX0tDwoKKre/o99L2fc+efJkzj//fG666SY2bNjABRdcUOm4ANdddx3Dhg0DYMSIEaVJJxEREREREWmYdL5+fOfrvlT22quuuop+/frx0Ucfcd555/Haa68xcOBAli9fzscff8zdd9/NxRdfzL333lvjfYMqlUpNHTSVyNDIcssiQyOZOmiq3/d98OBBXC4X0dHR7Ny5k08//bTW9zFgwADefvttwJlb6Suz+sMPP1BUVMT27dvZvHkzmzdv5u6772b27Nn079+f//73v2RkOP+QS6a/DR48mOnTpwPOl3n//v0EBQURGxvL+vXrKS4u5oMPPqg0rqysLNxuNwD//Oc/S5cPHjyY559/nqKionL7a9++PXFxcUybNo2xY8ee2IciIiIiIiIi9Z7O1ys3cODA0qvLrV27lp07d9K5c2c2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\n",
      "text/plain": [
       "<Figure size 1440x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "epochs = [i for i in range(20)]\n",
    "fig , ax = plt.subplots(1,2)\n",
    "train_acc = seq_model2.history['accuracy']\n",
    "train_loss = seq_model2.history['loss']\n",
    "val_acc = seq_model2.history['val_accuracy']\n",
    "val_loss = seq_model2.history['val_loss']\n",
    "fig.set_size_inches(20,10)\n",
    "\n",
    "ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')\n",
    "ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')\n",
    "ax[0].set_title('Training & Testing Accuracy')\n",
    "ax[0].legend()\n",
    "ax[0].set_xlabel(\"Epochs\")\n",
    "ax[0].set_ylabel(\"Accuracy\")\n",
    "\n",
    "ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')\n",
    "ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')\n",
    "ax[1].set_title('Training & Testing Loss')\n",
    "ax[1].legend()\n",
    "ax[1].set_xlabel(\"Epochs\")\n",
    "ax[1].set_ylabel(\"Loss\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train accuracy: 99.98409152030945\n",
      "Test accuracy: 99.87379312515259\n"
     ]
    }
   ],
   "source": [
    "train_gru_results = gru_model.evaluate(X_train_pad, y_train, verbose=0, batch_size=256)\n",
    "test_gru_results = gru_model.evaluate(X_test_pad, y_test, verbose=0, batch_size=256)\n",
    "print(\"Train accuracy: {}\".format(train_gru_results[1]*100))\n",
    "print(\"Test accuracy: {}\".format(test_gru_results[1]*100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuarcy: 99.87\n"
     ]
    }
   ],
   "source": [
    "y_pred = gru_model.predict_classes(X_test_pad)\n",
    "print(\"Accuarcy: {}\".format(round(accuracy_score(y_test, y_pred)*100,2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix: \n",
      " [[7062    7]\n",
      " [  10 6391]]\n"
     ]
    }
   ],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(\"Confusion Matrix: \\n\", cm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report: \n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00      7069\n",
      "           1       1.00      1.00      1.00      6401\n",
      "\n",
      "    accuracy                           1.00     13470\n",
      "   macro avg       1.00      1.00      1.00     13470\n",
      "weighted avg       1.00      1.00      1.00     13470\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Classification Report: \\n\", classification_report(y_test, y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
