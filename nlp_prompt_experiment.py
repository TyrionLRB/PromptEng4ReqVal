import csv
import nltk
import os
import random
import spacy
import time
import torch
import torchtext

import openai
import pandas as pd
from prettytable import PrettyTable

import utils

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 9000000
glove = torchtext.vocab.GloVe(name="6B", dim=50)
stopwords = nltk.corpus.stopwords.words('english')

# P2 Dataset
model_names_lst = ['2001-spacefractions']
#'2008-keepass', '1998-themas', '2003-qheadache'

cutoff_similarity_threshold = 0.85
cutoff_common_words_threshold = 250
num_pred_words_lst = [15]
wikidominer_depth_lst = [0]
masked_words = []

file = open("common_words_list.txt", "r", encoding="utf-8")
common_words_lst = file.read().split()
common_words_to_remove = common_words_lst[0:cutoff_common_words_threshold]
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


def query_gpt(masked_sentence, folder="prompt_patterns_complete"):
    prompts = utils.get_files_in_folder(folder)
    temperature = 0.4
    init_openai()

    reply = []
    for prompt in prompts:
        if "fewshot" in prompt:
            prompt = utils.read_file(prompt)
            prompt_fewshot = ("Here some examples from another requirements specification. "
                              "Q: The product shall {MASK} the data points in a scientifically correct manner. "
                              "A: The product shall plot the data points in a scientifically correct manner. "
                              "Q: The {MASK} axis should be labelled correctly according to the input from the data file. "
                              "A: The grid axis should be labelled correctly according to the input from the data file. "
                              "Q: The product should be able to {MASK} up to 2000 data points. "
                              "A: The product should be able to handle up to 2000 data points. "
                              "Q: The product should allow multiple points to clicked so that multiple names can be {MASK}. "
                              "A: The product should allow multiple points to clicked so that multiple names can be displayed. "
                              "Q: A double-click of the mouse over the data point should cause the application to display all the data point's {MASK}. "
                              "A: A double-click of the mouse over the data point should cause the application to display all the data point's details.")

            # ADD SAMPLED REQUIREMENTS TO PROMPT
            prompt = prompt + "\n" + masked_sentence
            # ASK COMPLETION
            reply = gpt4_prompt_fewshot(prompt, prompt_fewshot, engine="gpt-4-0125-preview", temperature=temperature)
        else:
            prompt = utils.read_file(prompt)

            # ADD SAMPLED REQUIREMENTS TO PROMPT
            prompt = prompt + "\n" + masked_sentence
            # ASK COMPLETION
            reply = gpt4_prompt(prompt, engine="gpt-4-0125-preview", temperature=temperature)

        return reply


# GloVe vectors used to calculate cosine similarity between two words
def cosine_sim(original_word, comparison_word):
    word1 = glove[original_word]
    word2 = glove[comparison_word]
    tensor_value = torch.cosine_similarity(word1.unsqueeze(0), word2.unsqueeze(0))
    return tensor_value.item()


# accuracy = Number of matched terms / Number of terms in Predictions not in First Half
def compute_accuracy(matched_words, predictions_not_in_first_half):
    if matched_words == 0:
        return 0
    return ((matched_words / predictions_not_in_first_half) * 100)


# coverage = Number of matched terms / Number of all terms in second half of text
def compute_coverage(matched_words, updated_second_half_words_without_duplicates):
    if matched_words == 0:
        return 0
    return ((matched_words / len(updated_second_half_words_without_duplicates)) * 100)


# Extract individual words from each sentence
def extract_words(sentence):
    doc = nlp(sentence)
    # create list of tokens from sentence
    words_lst = [token.text for token in doc]
    word_pos_tags = [token.pos_ for token in doc]
    return words_lst, word_pos_tags


def lst_to_pdf(lst, model_name):
    # Create new input document to feed  WikiDoMiner
    file_name = model_name + ".txt"
    file = open(file_name, "w", encoding="utf-8")
    for i in lst:
        file.writelines(i)
        file.writelines("\n")
    file.close()
    # Output text file as BERT fine-tuning text
    return file_name


# Extract individual sentences from text
def extract_sentences_from_file(text):
    file_sentences = []
    for paragraph in text:
        sentences = [sentence for sentence in paragraph.split('.') if sentence != '' and len(sentence.split()) > 3]
        sentences = list(map(str.strip, sentences))
        file_sentences.extend(sentences)
    return file_sentences


# Extract and lemmatize individual words from each sentence
def extract_lemmatize_words(sentence):
    doc = nlp(sentence)
    # create list of tokens from sentence
    words_lst = [token.text for token in doc]
    token_lst = [token for token in doc]
    # check if tokens are NOUNS/VERBS and lemmatize if true
    for token in token_lst:
        if token.pos_ == 'NOUN' or token.pos_ == 'VERB':
            words_lst[token_lst.index(token)] = token.lemma_
    return words_lst


def init_openai(key=None):
    if not key:
        key = open("openai_key.txt", 'r').read()
    openai.api_key = key


def gpt4_prompt(prompt, engine="gpt-4-0125-preview", max_tokens=500, temperature=0.4, maxiter=1):
    while maxiter > 0:
        try:
            response = openai.chat.completions.create(
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature)

            return response.choices[0].message.content.strip()
        except Exception as e:
            maxiter -= 1
            print(f"API not ready. Error: {str(e)}")
            print("Trying again:", maxiter, "attempts left")
            time.sleep(20)
    return "API not ready"


def gpt4_prompt_fewshot(prompt, prompt_fewshot, engine="gpt-4-0125-preview", max_tokens=500, temperature=0.4,
                        maxiter=1):
    while maxiter > 0:
        try:
            response = openai.chat.completions.create(
                model=engine,
                messages=[{"role": "user", "content": prompt},
                          {"role": "assistant", "content": prompt_fewshot}],
                max_tokens=max_tokens,
                temperature=temperature)

            return response.choices[0].message.content.strip()
        except Exception as e:
            maxiter -= 1
            print(f"API not ready. Error: {str(e)}")
            print("Trying again:", maxiter, "attempts left")
            time.sleep(20)
    return "API not ready"


# Match predicted words to second half of text
def match_against_2nd_half(terms, updated_second_half_words_without_duplicates, all_first_half_words_lemmatized):
    predictions_not_in_first_half = 0
    all_matched_terms = []
    novel_matched_terms = []
    duplicate_free_matched_terms = []

    for item in terms:
        # Ensure the predicted word is lemmatized
        doc = nlp(item)
        if doc[0].pos_ == 'NOUN' or doc[0].pos_ == 'VERB':
            item = doc[0].lemma_
            if item not in all_first_half_words_lemmatized:
                predictions_not_in_first_half += 1
                # Search for predicted words that are semantically close to words that appear in second half
                for word in updated_second_half_words_without_duplicates:
                    distance = cosine_sim(item.casefold(), word.casefold())
                    # Write out words that exceed the cosine sim. requirement and whose lemmatized form does not appear in 1st half of text
                    if distance >= cutoff_similarity_threshold:
                        all_matched_terms.append(item)
                        # Assign only one prediction to each word in 2nd half
                        if word not in novel_matched_terms:
                            novel_matched_terms.append(word)
                        # Filter out duplicate predictions
                        if item not in duplicate_free_matched_terms:
                            duplicate_free_matched_terms.append(item)
                            break
    return predictions_not_in_first_half, len(all_matched_terms), len(novel_matched_terms), len(
        duplicate_free_matched_terms)


# Funktion zum Maskieren von Wörtern basierend auf POS-Tags
def mask_sentence(words_lst, word_pos_tags, mask_pos_list):
    masked_sentences = []
    check_used = []
    loopings = word_pos_tags.count(mask_pos_list[0])
    i = 0
    while i < loopings:
        masked_words = []
        flg = ""
        for word, pos in zip(words_lst, word_pos_tags):
            if pos in mask_pos_list and word not in check_used and flg != "x":
                masked_words.append('[MASK]')
                check_used.append(word)
                flg = "x"
            else:
                masked_words.append(word)
        i += 1
        masked_sentences.append(' '.join(masked_words))
    return masked_sentences


output_table = PrettyTable(["Predictions/Mask", "Total Num. of predictions", "Num. of non-duplicate predictions",
                            "Num. of predictions not in 1st half", "Num. of total matched terms (with duplicates)",
                            "Num. of total matched terms (without duplicates)", "Num. of novel matched terms",
                            "Num. of non-duplicate terms in 2nd half", "Accuracy", "Coverage"])

home_path = os.getcwd()
for model_name in model_names_lst:
    #    output_table = PrettyTable(["Predictions/Mask", "Total Num. of predictions", "Num. of non-duplicate predictions", "Num. of predictions not in 1st half", "Num. of total matched terms (with duplicates)", "Num. of total matched terms (without duplicates)", "Num. of novel matched terms", "Num. of non-duplicate terms in 2nd half", "Accuracy", "Coverage"])
    # Anpassung Herkunft Datei
    dataset = home_path + "/data/" + model_name + ".txt"
    with open(dataset, "rt", encoding="UTF-8") as data:
        text = data.read().split('\n')
    text_sentences = extract_sentences_from_file(text)
    data.close()

    #    wir arbeiten hier nur mit 15
    #    for num_pred_words in num_pred_words_lst:
    random.shuffle(text_sentences)
    first_half_text = text_sentences[:len(text_sentences) // 2]
    second_half_text = text_sentences[len(text_sentences) // 2:]

    # Extract and lemmatize words from first half of text
    all_first_half_words_lemmatized = []
    for sentence in first_half_text:
        all_first_half_words_lemmatized.extend(extract_lemmatize_words(sentence))
    print(len(all_first_half_words_lemmatized))
    # Extract and lemmatize words from second half of text
    all_second_half_words_lemmatized = []
    for sentence in second_half_text:
        all_second_half_words_lemmatized.extend(extract_lemmatize_words(sentence))
    print(len(all_second_half_words_lemmatized))

    wikidominer_title = '"' + lst_to_pdf(first_half_text, model_name) + '"'
    corpus_folder = model_name + "_" + str(num_pred_words_lst) + "_Predictions/Mask_Corpus"
    merged_corpus = 'all_merged_files'
    #        bert_training_corpus = corpus_folder+"/"+merged_corpus
    return_code = os.system(
        "python WikiDoMiner.py --doc {a} --output-dir {b} --wiki-depth {c}".format(a=wikidominer_title, b=corpus_folder,
                                                                                   c=0))
    print(f"Return code: {return_code}")

    # Merge individual WikiDoMiner files into single corpus
    os.chdir(corpus_folder)
    corpus_files_name = os.listdir()

    merge_lines = ""
    #        os.system("cat * > {}".format(merged_corpus))
    for file_corpus in corpus_files_name:
        if file_corpus != merged_corpus:
            with open(file_corpus, "r", encoding="utf-8") as merge_file:
                merge_lines = merge_lines + merge_file.read() + "/n"

    with open(merged_corpus, "w", encoding="utf-8") as write_merged_corpus:
        write_merged_corpus = write_merged_corpus.write(merge_lines)

    # Read in merged WikiDoMiner corpus file and lemmatize words for bucket creation
    with open(merged_corpus, "r", encoding="utf-8") as corpus_file:
        corpus = corpus_file.read()
    corpus_text = corpus.split('\n')
    corpus_sentences = extract_sentences_from_file(corpus_text)
    lemmatized_corpus = []
    for sentence in corpus_sentences:
        lemmatized_corpus.extend(extract_lemmatize_words(sentence))

    # Remove stopwords from corpus
    updated_corpus = [x.lower() for x in lemmatized_corpus if x not in stopwords]

    # Process documents in corpus for TFIDF - Buckets werden nicht verwendet
    for file in corpus_files_name:
        # Read in document and clean text by lemmatizing and removing stopwords
        with open(file, "r", encoding="utf-8") as corpus_file:
            file_text = corpus_file.read()
            lemmatized_text = extract_lemmatize_words(file_text)
            updated_text = [x.lower() for x in lemmatized_text if x not in stopwords]
            cleaned_text = " ".join(updated_text)
        os.remove(file)
        # Write out cleaned text to same filename
        # Sicherstellen, dass die Dateien in UTF-8 kodiert sind, da es ansonsten Fehler gibt beim tfidf_vectorizer
        with open(file, 'w', encoding='utf8') as w:
            w.write(cleaned_text)

    #   Keine tfidf Berechnung, da dies in der Baseline erfolgt. Keine Relevant hierfür,
    #   da wir 15 Vorschläge vom GPT erhalten und nicht die wahrscheinlichsten Vorschläge filtern

    os.chdir(home_path)

    # Create file containing output values at each interval of cosine sim. and common word removal
    features_lst = []
    prediction_class_lst = []
    predicted_words_lemmatized = []
    df_rows_lst = []
    #        temploop_breaker_sentence = 0

    for sentence in first_half_text:
        # Create list of tokens and corresponding POS tag from sentence
        #            temploop_breaker_sentence = temploop_breaker_sentence + 1
        sentence_words = [x.strip(' ') for x in sentence.split()]
        words_lst, word_pos_tags = extract_words(" ".join(sentence_words))

        # Sätze maskieren für jeweils Verben und Nomen - Für jedes maskierte Wort ein eigener Satz
        masked_verbs_sentences = mask_sentence(words_lst, word_pos_tags, ['VERB'])
        masked_nouns_sentences = mask_sentence(words_lst, word_pos_tags, ['NOUN'])

        # Tokenisiere den Satz mit maskierten Verben
        #            tokens_masked_verbs = tokenize_sentence(masked_verbs_sentences)
        #            tokens_masked_nouns = tokenize_sentence(masked_nouns_sentences)

        #            temploop_breaker = 0

        # Loopen über die Sätze mit den jeweils maskierten Wörtern
        for masked_verbs_sentence in masked_verbs_sentences:
            # Aufruf von GPT zur Vorschlagsgenerierung für Verben
            predicted_words = query_gpt(masked_verbs_sentence, "prompt_patterns_complete")
            utils.write_text_file(predicted_words)
            predicted_words = predicted_words.split()
            masked_verbs_sentence_mask_index = masked_verbs_sentence.split()
            for predicted_word in predicted_words:
                if "[MASK]" in masked_verbs_sentence_mask_index:
                    word = words_lst[masked_verbs_sentence_mask_index.index("[MASK]")]
                    # Wenn kein Stoppwort
                    if predicted_word not in stopwords:
                        # Vorschlag speichern und dataframe Eintrag generieren
                        doc = nlp(predicted_word)
                        new_row = {'Actual_Word': word, 'Predicted_Word': predicted_word,
                                   'Predicted_Word_Lemmatized': doc[0].lemma_, 'Token_Weight': "tbd",
                                   'Mask_Index': masked_verbs_sentence.find("[MASK]"),
                                   'Masked_Sentence': masked_verbs_sentence, 'Sentence': words_lst}
                        df_rows_lst.append(new_row)
                        predicted_words_lemmatized.append(doc[0].lemma_)
        #                temploop_breaker = temploop_breaker + 1
        #                if temploop_breaker <= 10:
        #                    break

        #            temploop_breaker = 0

        # Loopen über die Sätze mit den jeweils maskierten Wörtern
        for masked_nouns_sentence in masked_nouns_sentences:
            # Aufruf von GPT zur Vorschlagsgenerierung für Nomen
            predicted_words = query_gpt(masked_nouns_sentence, "prompt_patterns_complete")
            utils.write_text_file(predicted_words)
            predicted_words = predicted_words.split()
            masked_nouns_sentence_mask_index = masked_nouns_sentence.split()
            for predicted_word in predicted_words:
                if "[MASK]" in masked_nouns_sentence_mask_index:
                    word = words_lst[masked_nouns_sentence_mask_index.index("[MASK]")]
                    if predicted_word not in stopwords:
                        doc = nlp(predicted_word)
                        new_row = {'Actual_Word': word, 'Predicted_Word': predicted_word,
                                   'Predicted_Word_Lemmatized': doc[0].lemma_, 'Token_Weight': "tbd",
                                   'Mask_Index': masked_nouns_sentence.find("[MASK]"),
                                   'Masked_Sentence': masked_nouns_sentence, 'Sentence': words_lst}
                        df_rows_lst.append(new_row)
                        predicted_words_lemmatized.append(doc[0].lemma_)
    #                temploop_breaker = temploop_breaker + 1
    #                if temploop_breaker <= 10:
    #                    break

    #           if temploop_breaker_sentence <= 10:
    #                break

    # Remove stopwords and duplicate words from lists
    second_half_words_lowercase_without_duplicates = [x.lower() for x in list(set(all_second_half_words_lemmatized))]
    updated_second_half_words_without_duplicates = list(
        set(second_half_words_lowercase_without_duplicates) - set(stopwords))
    updated_novel_second_half_words = list(
        set(updated_second_half_words_without_duplicates) - set(all_first_half_words_lemmatized))

    updated_predicted_words_without_duplicates = [x.lower() for x in list(set(predicted_words_lemmatized))]
    updated_predicted_words_with_duplicates = [x.lower() for x in predicted_words_lemmatized]

    # Write out to file the predicted words not in first half of text but appear
    # semantically close to words in second half list of words
    predictions_not_in_first_half, all_matched_terms, novel_matched_terms, duplicate_free_matched_terms \
        = match_against_2nd_half(updated_predicted_words_without_duplicates,
                                 updated_novel_second_half_words, all_first_half_words_lemmatized)

    # accuracy = Number of matched terms / Number of predicted terms not in 1st half
    accuracy = compute_accuracy(duplicate_free_matched_terms, predictions_not_in_first_half)

    # coverage = Number of matched terms / Number of all terms in second half of text
    coverage = compute_coverage(novel_matched_terms, updated_novel_second_half_words)

    # Create dataframe and remove duplicate tokens for feature matrix
    df_with_duplicate_predictions = pd.DataFrame(df_rows_lst)
    df_without_duplicates = df_with_duplicate_predictions.drop_duplicates(subset=['Predicted_Word'])

    print("Model Name: ", model_name)
    print("\nNumber of predicted words per mask: ", num_pred_words_lst)
    print("Total number of predictions: ", len(predicted_words_lemmatized))
    print("Number of non-duplicate predictions: ", len(updated_predicted_words_without_duplicates))
    print("Number of predictions not in 1st half: ", predictions_not_in_first_half)
    print("Number of total matched terms (with duplicates): ", all_matched_terms)
    print("Number of total matched terms (without duplicates): ", duplicate_free_matched_terms)
    print("Number of novel matched terms: ", novel_matched_terms)
    print("Number of non-duplicate terms in second half of text: ", len(updated_novel_second_half_words))
    print("Accuracy: ", accuracy, " | Coverage: ", coverage)
    print("----------------------------------------------------------------------------------------")

    output_table.add_row([num_pred_words_lst, len(predicted_words_lemmatized),
                          len(updated_predicted_words_without_duplicates),
                          predictions_not_in_first_half, all_matched_terms, duplicate_free_matched_terms,
                          novel_matched_terms, len(updated_novel_second_half_words), accuracy, coverage])

    details_matched_pred_words_file_name = (model_name + "_" + str(num_pred_words_lst) + "_Predictions/" + model_name
                                            + "det_matched_pred_Words.txt")

    with (open(details_matched_pred_words_file_name, 'w', newline='', encoding="UTF-8")
          as details_matched_pred_words_file):
        details_matched_pred_words_writer = csv.writer(details_matched_pred_words_file)
        for i, row in df_without_duplicates.iterrows():
            details_matched_pred_words_writer.writerow(row.values.tolist())

    p1_file_name = model_name + "_" + str(num_pred_words_lst) + "_Predictions/" + model_name + "_P1_Words.txt"
    p1_file = open(p1_file_name, 'w', encoding="UTF-8")
    p1_writer = csv.writer(p1_file)
    for word in all_first_half_words_lemmatized:
        p1_writer.writerow([word])

    # Write out P1 terms
    p2_file_name = model_name + "_" + str(num_pred_words_lst) + "_Predictions/" + model_name + "_P2_Words.txt"
    p2_file = open(p2_file_name, 'w', encoding="UTF-8")
    p2_writer = csv.writer(p2_file)
    for word in all_second_half_words_lemmatized:
        p2_writer.writerow([word])

    # Write output_table and yes class prediction values to file
    output_table_file_name = model_name + "_Output_Table.txt"
    with open(output_table_file_name, 'w', encoding="UTF-8") as w:
        w.write(str(output_table))
    print("\nOutput written out to:", output_table_file_name)
