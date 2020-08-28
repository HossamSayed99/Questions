# Questions

An AI that answers questions based on the corpus of data provided.

## Usage 

    $ python questions.py corpus
    Query: What are the types of supervised learning?
    Types of supervised learning algorithms include Active learning , classification and regression.

    $ python questions.py corpus
    Query: When was Python 3.0 released?
    Python 3.0 was released on 3 December 2008.

    $ python questions.py corpus
    Query: How do neurons connect in a neural network?
    Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers.

## Details
This question answering system performs two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined. <br>
Most relevant documents and passages are determined through using a combination of **inverse document frequency and a query term density measure**.

## Required dependencies

Inside of the `questions` directory, `run pip3 install -r requirements.txt` to install this project’s dependency: `nltk` for natural language processing.