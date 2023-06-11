import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import keras
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import classification_report

dataset_to_max_length = {
    "imdb": 512,
    "dbpedia": 512,
    "ag_news": 64,
}

dataset_to_num_labels = {"imdb": 2, "dbpedia": 9, "ag_news": 4}

STOPWORDS = set(stopwords.words("english"))
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")

import cufflinks
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme="pearl")
import os


def get_model(X, MAX_NB_WORDS, EMBEDDING_DIM, num_labels):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_labels, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )
    print(model.summary())

    return model


def clean_text(text):
    """
    text: a string

    return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(
        " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(
        "", text
    )  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace("x", "")
    #    text = re.sub(r'\W+', '', text)
    text = " ".join(
        word for word in text.split() if word not in STOPWORDS
    )  # remove stopwors from text
    return text


import keras.backend as K


def size(model):  # Compute number of params in a model (the actual number of floats)
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


def load_dataset(dataset):
    train_df = pd.read_csv(
        os.path.join("../datasets", f"{dataset}_dataset", "train.csv")
    )
    train_df["text"] = train_df["text"].apply(clean_text)

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = dataset_to_max_length[dataset]
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(
        num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
    )
    tokenizer.fit_on_texts(train_df["text"].values)
    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    def tokenize_df(df):
        X = tokenizer.texts_to_sequences(df["text"].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        Y = pd.get_dummies(df["label"]).values
        return X, Y

    X_train, Y_train = tokenize_df(train_df)

    test_df = pd.read_csv(os.path.join("../datasets", f"{dataset}_dataset", "test.csv"))
    test_df["text"] = test_df["text"].apply(clean_text)
    X_test, Y_test = tokenize_df(test_df)

    return X_train, Y_train, X_test, Y_test, MAX_NB_WORDS, EMBEDDING_DIM


def main(args):
    X_train, Y_train, X_test, Y_test, MAX_NB_WORDS, EMBEDDING_DIM = load_dataset(
        args.dataset
    )

    model = get_model(
        X_train, MAX_NB_WORDS, EMBEDDING_DIM, dataset_to_num_labels[args.dataset]
    )
    print("Number of params:", size(model))
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=args.model_dir, save_weights_only=True, save_best_only=True, verbose=1
    )

    if args.mode == "train":
        print("Training the model...")
        history = model.fit(
            X_train,
            Y_train,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001),
                cp_callback,
            ],
        )
    else:
        model.load_weights(args.model_dir)

    print("Evaluating the model on the test set...")
    accr = model.evaluate(X_test, Y_test)
    all_predictions = model.predict(X_test)
    print("Test set\n  Loss: {:0.3f}".format(accr[0]))
    print("The classification report for the model is:")
    print(
        classification_report(
            np.argmax(Y_test, axis=1), np.argmax(all_predictions, axis=1)
        )
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )

    argparser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )

    argparser.add_argument("--num_epochs", type=int, default=3)
    argparser.add_argument(
        "--mode",
        type=str,
        required=True,
    )

    args = argparser.parse_args()

    main(args)
