from statistics import mean

import nltk
import numpy as np
import os.path
import pandas as pd
import spacy
import torch
import torchtext
from nltk.corpus import wordnet
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer

model_names_lst = ['2008-keepass']
# , '2008-keepass','1998-themas', '2003-qheadache'

cutoff_common_words_threshold = 250
nlp = spacy.load("en_core_web_sm")
glove = torchtext.vocab.GloVe(name="6B", dim=50)
stopwords = nltk.corpus.stopwords.words('english')

file = open("common_words_list.txt", "r", encoding="UTF-8")
common_words = file.read()
common_words_to_remove = common_words.split()[0:cutoff_common_words_threshold]
for i in common_words_to_remove:
    stopwords.append(i)

# Remove additional stopwords from text
other_stopwords = (
'-', '।', '|', '!', '?', ',', '.', '...', ':', ';', '@', '$', '%', '^', '&', '*', '/', '(', ')', '<', '>', '[', ']',
'{', '}', '~', '`', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th',
'8th', '9th', "#", "+", "=", '\t', '\n', '\n\n', "s", 'the', 'as', 'are', 'own', 'in', 'ii', 'iii', 'iv', 'v', 'a',
'ed', 'first', 'second', 'third', 'fourth', 'fifth')
for i in other_stopwords:
    stopwords.append(i)


# GloVe vectors used to calculate cosine similarity between two words
def cosine_sim(original_word, comparison_word):
    first_word = original_word.casefold()
    second_word = comparison_word.casefold()
    word1 = glove[first_word]
    word2 = glove[second_word]
    tensor_value = torch.cosine_similarity(word1.unsqueeze(0), word2.unsqueeze(0))
    return tensor_value.item()


# accuracy = Number of matched terms / Number of all predicted terms
def compute_accuracy(yes_predictions, all_predictions):
    if len(yes_predictions) == 0 or len(all_predictions) == 0:
        return 0
    return ((len(yes_predictions) / len(all_predictions)) * 100)


# Coverage = Number of P2 matched terms / Number of all terms in P2
def compute_coverage(coverage_lst, p2_words_lst):
    if len(coverage_lst) == 0 or len(p2_words_lst) == 0:
        return 0
    return ((len(coverage_lst) / len(p2_words_lst)) * 100)


# Extract individual sentences from text
def extract_sentences_from_file(text):
    file_sentences = []
    for paragraph in text:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != '' and len(sentence.split()) > 3]
        sentences = list(map(str.strip, sentences))
        file_sentences.extend(sentences)
    return file_sentences


def baseline_1(p1_words_lst):
    all_common_words = common_words.split()[cutoff_common_words_threshold:2000]
    b1_common_words = list(set(all_common_words) - set(stopwords))
    novel_common_word_predictions = list(set(b1_common_words) - set(p1_words_lst))
    return novel_common_word_predictions


def baseline_2(model_name, p1_words_lst, iteration):
    os.chdir(model_name + "_[15]_Predictions/Mask_Corpus")
    corpus_files_name = os.listdir()
    corpus_files_name.remove('all_merged_files')
    tfidf_vectorizer = TfidfVectorizer(input='filename')
    tfidf_total = tfidf_vectorizer.fit_transform(corpus_files_name)
    tfidf_df = pd.DataFrame(tfidf_total.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df.replace(0, np.nan, inplace=True)
    mean_series = tfidf_df.mean(axis=0, skipna=True)
    mean_df = mean_series.to_frame().T  # Transponieren, um eine Zeile zu erhalten
    mean_df.index = ['Mean']  # Setze den Indexnamen auf 'Mean'
    # Hänge die Zeile an tfidf_df an
    tfidf_df = pd.concat([tfidf_df, mean_df], ignore_index=False)

    # Berechne das Maximum jeder Spalte und konvertiere die Serie in einen DataFrame
    max_series = tfidf_df.max(axis=0, skipna=True)
    max_df = max_series.to_frame().T  # Transponieren, um eine Zeile zu erhalten
    max_df.index = ['Max']  # Setze den Indexnamen auf 'Max'
    # Hänge die Zeile an tfidf_df an
    tfidf_df = pd.concat([tfidf_df, max_df], ignore_index=False)

    tfidf_df = tfidf_df.loc[:, tfidf_df.max().sort_values(ascending=False).index]
    #    top_x_words = tfidf_df.size * 0.8
    #    cutoff_word = tfidf_df.columns[round(top_x_words)]
    #.loc[:, : cutoff_word]
    all_tfidf_predictions = tfidf_df

    tfidf_predictions = list(set(all_tfidf_predictions.columns.values.tolist()) - set(stopwords))
    novel_tfidf_predictions = list(set(tfidf_predictions) - set(p1_words_lst))
    os.chdir(home_path)
    return novel_tfidf_predictions


def baseline_3(p1_words_lst):
    baseline_3_lst = []
    for word in p1_words_lst:
        num_synonyms = 0
        for syn in wordnet.synsets(word):
            for name in syn.lemma_names():
                if name not in baseline_3_lst and num_synonyms < 2:
                    num_synonyms = 1 + num_synonyms
                    baseline_3_lst.append(name)

    synonym_predictions = list(set(baseline_3_lst) - set(stopwords))
    novel_synonym_predictions = list(set(synonym_predictions) - set(p1_words_lst))
    return novel_synonym_predictions


def preds_appear_p2(all_words, p2_words_lst):
    novel_words = []
    coverage_words = []
    duplicate_free_matched_terms = []
    for word in all_words:
        doc = nlp(word)
        if doc[0].pos_ == 'NOUN' or doc[0].pos_ == 'VERB':
            for item in p2_words_lst:
                cosine = cosine_sim(word.casefold(), item.casefold())
                if cosine >= 0.85 and word:
                    if word not in novel_words:
                        novel_words.append(word)
                    if item not in coverage_words:
                        coverage_words.append(item)
    return novel_words, coverage_words


home_path = os.getcwd()
for model_name in model_names_lst:
    output_table = PrettyTable(["Iteration", "Baseline 1 - Accuracy", "Baseline 1 - Coverage",
                                "Baseline 1 - Num, of non-duplicate predictions", "Baseline 2 - Accuracy",
                                "Baseline 2 - Coverage", "Baseline 2 - Num, of non-duplicate predictions",
                                "Baseline 3 - Accuracy", "Baseline 3 - Coverage",
                                "Baseline 3 - Num, of non-duplicate predictions"])

    accuracy_lst = []
    coverage_lst = []

    baseline_1_accuracy_lst = []
    baseline_1_coverage_lst = []
    baseline_1_novel_common_word_predictions = []

    baseline_2_accuracy_lst = []
    baseline_2_coverage_lst = []
    baseline_2_novel_tfidf_predictions = []

    baseline_3_accuracy_lst = []
    baseline_3_coverage_lst = []
    baseline_3_novel_synonym_predictions = []

    for i in range(5):
        p1_text = model_name + "_[15]_Predictions/" + model_name + "_" + str(i) + "_P1_Words.txt"
        p1_file = open(p1_text, "r", encoding="UTF-8")
        p1_words = p1_file.read()
        p1_words_lst = [x.strip(' ') for x in p1_words.split()]

        p2_text = model_name + "_[15]_Predictions/" + model_name + "_" + str(i) + "_P2_Words.txt"
        p2_file = open(p2_text, "r", encoding="UTF-8")
        p2_words = p2_file.read()
        p2_words_lst = [x.strip(' ') for x in p2_words.split()]

        novel_common_word_predictions = baseline_1(p1_words_lst)
        b1_accuracy_words, b1_coverage_words = preds_appear_p2(novel_common_word_predictions, p2_words_lst)
        baseline_1_accuracy = compute_accuracy(b1_accuracy_words, novel_common_word_predictions)
        baseline_1_coverage = compute_coverage(b1_coverage_words, p2_words_lst)
        baseline_1_accuracy_lst.append(baseline_1_accuracy)
        baseline_1_coverage_lst.append(baseline_1_coverage)
        baseline_1_novel_common_word_predictions.append(len(novel_common_word_predictions))

        novel_tfidf_predictions = baseline_2(model_name, p1_words_lst, i)
        b2_accuracy_words, b2_coverage_words = preds_appear_p2(novel_tfidf_predictions, p2_words_lst)
        baseline_2_accuracy = compute_accuracy(b2_accuracy_words, novel_tfidf_predictions)
        baseline_2_coverage = compute_coverage(b2_coverage_words, p2_words_lst)
        baseline_2_accuracy_lst.append(baseline_2_accuracy)
        baseline_2_coverage_lst.append(baseline_2_coverage)
        baseline_2_novel_tfidf_predictions.append(len(novel_tfidf_predictions))

        novel_synonym_predictions = baseline_3(p1_words_lst)
        b3_accuracy_words, b3_coverage_words = preds_appear_p2(novel_synonym_predictions, p2_words_lst)
        baseline_3_accuracy = compute_accuracy(b3_accuracy_words, novel_synonym_predictions)
        baseline_3_coverage = compute_coverage(b3_coverage_words, p2_words_lst)
        baseline_3_accuracy_lst.append(baseline_3_accuracy)
        baseline_3_coverage_lst.append(baseline_3_coverage)
        baseline_3_novel_synonym_predictions.append(len(novel_synonym_predictions))

        output_table.add_row(
            [i, baseline_1_accuracy, baseline_1_coverage, len(novel_common_word_predictions),
             baseline_2_accuracy, baseline_2_coverage, len(novel_tfidf_predictions),
             baseline_3_accuracy, baseline_3_coverage, len(novel_synonym_predictions)])

    b1_avg_accuracy = round(mean(baseline_1_accuracy_lst), 2)
    b1_avg_coverage = round(mean(baseline_1_coverage_lst), 2)
    b1_avg_num_pred = round(mean(baseline_1_novel_common_word_predictions), 2)

    b2_avg_accuracy = round(mean(baseline_2_accuracy_lst), 2)
    b2_avg_coverage = round(mean(baseline_2_coverage_lst), 2)
    b2_avg_num_pred = round(mean(baseline_2_novel_tfidf_predictions), 2)

    b3_avg_accuracy = round(mean(baseline_3_accuracy_lst), 2)
    b3_avg_coverage = round(mean(baseline_3_coverage_lst), 2)
    b3_avg_num_pred = round(mean(baseline_3_novel_synonym_predictions), 2)

    output_table.add_row(['-------', '-------', '-------', '-------', '-------', '-------', '-------', '-------',
                          '-------', '-------'])
    output_table.add_row(
        ['Average', b1_avg_accuracy, b1_avg_coverage, b1_avg_num_pred,
         b2_avg_accuracy, b2_avg_coverage, b2_avg_num_pred, b3_avg_accuracy,
         b3_avg_coverage, b3_avg_num_pred])

    output_file_name = "Baselines_" + model_name + ".txt"
    with open(output_file_name, 'w', encoding="UTF-8") as w:
        w.write(str(output_table))

    print("\n ", model_name)
    print("B1 Accuracy: ", b1_avg_accuracy, " | B1 Coverage: ", b1_avg_coverage)
    print("B1 Num Pred: ", len(novel_common_word_predictions))
    print("B2 Accuracy: ", b2_avg_accuracy, " | B2 Coverage: ", b2_avg_coverage)
    print("B2 Num Pred: ", len(novel_tfidf_predictions))
    print("B3 Accuracy: ", b3_avg_accuracy, " | B3 Coverage: ", b3_avg_coverage)
    print("B3 Num Pred: ", len(novel_synonym_predictions))
    print("Output written out to:", output_file_name)
