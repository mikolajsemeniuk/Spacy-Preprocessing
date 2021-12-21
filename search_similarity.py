import pandas as pd
import spacy

from pandas import DataFrame, Series

pd.set_option('display.max_rows', 100)


dataset: DataFrame = \
    pd.read_csv('polish_sentiment_dataset.csv')

dataset = dataset.convert_dtypes()
dataset = dataset[dataset.description.notna()]
dataset = dataset[dataset.rate.notna()]
dataset = dataset[dataset.rate != 0]


nlp = spacy.load("pl_core_news_lg")


def get_similar_sentences(sentences: Series, search_sentence: str):
    sorted_sentences_by_similarity = []
    search_vector = nlp(search_sentence)
    search_vector_without_stopwords = nlp(' '.join([str(word) for word in search_vector if not word.is_stop]))

    for sentence in sentences:
        sentence_vector = nlp(sentence)
        sentence_vector_without_stopwords = nlp(' '.join([str(word) for word in sentence_vector if not word.is_stop]))
        sorted_sentences_by_similarity.append((sentence, search_vector_without_stopwords.similarity(sentence_vector_without_stopwords)))
    
    similarities = \
        pd.DataFrame(sorted_sentences_by_similarity, columns = ['sentence', 'similarity'])

    return similarities.sort_values(by = ['similarity'], ascending = False)

x = get_similar_sentences(dataset.description.iloc[1:100], dataset.description.iloc[0])

print(f'\nwyraz do wyszukania: {dataset.description.iloc[4]}\n')
print(x.head(100))
