## Big files of our CIL project
https://polybox.ethz.ch/index.php/s/Tb0QWEKEK9Bhiqy

## Twitter  Datasets

Download the tweet datasets from here:
https://polybox.ethz.ch/index.php/s/pp6Mzg7BwcXVG5z


The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

- vocab.txt vocab.pkl vocab_cut.txt
- cooc.pkl: cooccurance matrix 

## Build the Co-occurence Matrix (already in datasets but feel free to rerun)

To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py

##  Template for Glove Question (already in polybox embeddings)

Your task is to fill in the SGD updates to the template
glove_template.py

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt

Note: std_glove_embeddings.npz is trained with the glove_solution.py provided by the course group. We should use it in part of the baseline.
can be downloaded from:
https://polybox.ethz.ch/index.php/s/JQ8awPuk5tdrp5A


##  available embedding so far
standard glove embedding (provided by the TA group)
https://polybox.ethz.ch/index.php/s/JQ8awPuk5tdrp5A