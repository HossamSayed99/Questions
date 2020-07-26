import nltk
import sys
import os
import string
import math
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    ret = dict()
    # Looping over all files in a directory
    for file in os.listdir(directory):
        # opening each file and reading it
        f = open(os.path.join(directory, file), encoding='utf8')
        ret[file] = f.read()

    return ret


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Converting all words to lower case
    document = document.lower()
    # tokeniizing the string to words
    words = nltk.tokenize.word_tokenize(document)
    final_words = []
    # Add the word to the string if it is not a punctuatuion or an English stop word
    for word in words:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            final_words.append(word)

    return final_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # The dictionary to be returned
    ret = dict()
    # A dictionary that will map each word to the number of its occurences
    frequency = dict()
    
    total_documents = len(documents)
    # Counting the occurences of each word
    for document in documents:
        for word in documents[document]:
            if word not in frequency:
                frequency[word] = {document}
            else:
                frequency[word].add(document)

    # Calculating the idf for each word
    for word in frequency:
        ret[word] = math.log(total_documents / len(frequency[word]))

    return ret


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # A dictionary that maps each file to the sum of tf-ifds of the words in it and appears in the query
    files_to_tfifd = dict()

    # Loop over each word in the query
    for word in query:
        # Loop over all files for each word
        for file in files:
            # if the word exist in the file
            if word in files[file]:
                # increase its total tf-ifd value
                files_to_tfifd[file] = files_to_tfifd.get(file, 0) + (idfs[word] * files[file].count(word))

    # Sorta files in a decscending order according to the value of the total tf-ifdf
    files_sorted = sorted(files.keys(), key=lambda file: files_to_tfifd.get(file, 0), reverse=True)

    return files_sorted[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # A dictionary that maps the senetence to the query term density
    sentence_to_query_term_density = dict()

    # A dictionary that maps the sentence to the total sum of idfs of the words it has in common with the query

    sentence_to_totaldfs = dict()

    for word in query:
        for sentence in sentences:
            # If a word is common between the query and teh senetence, increase the value of the totaldfs for that sentence
            if word in sentences[sentence]:
                sentence_to_totaldfs[sentence] = sentence_to_totaldfs.get(sentence, 0) + idfs[word]
                # sentence_to_query_term_density[sentence] = sentence_to_query_term_density.get(sentence, 0) + 1
    
    for sentence in sentences:
        sentence_to_query_term_density[sentence] = len([w for w in sentences[sentence] if w in query]) / len(sentences[sentence])

    # for sentence in sentence_to_totaldfs:
    #     sentence_to_totaldfs[sentence] /= len(sentences[sentence])

    # Sort all sentences accordign to the sum of idf of the common words beteen it and the query 
    # and then the query term density
    sentences_sorted = sorted(sentences.keys(), 
                              key=lambda k: (sentence_to_totaldfs.get(k, 0), sentence_to_query_term_density.get(k, 0)), 
                              reverse=True)
    # Returning best n sentences
    return sentences_sorted[:n]


if __name__ == "__main__":
    main()
