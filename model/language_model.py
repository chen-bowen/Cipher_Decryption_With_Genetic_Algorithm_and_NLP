import pandas as pd
import numpy as np
import requests
import re
import os


class LanguageModel:
    """Build a language model using Moby Dick text corpus"""

    def __init__(self):
        self.build()

    def __load_data(self):
        """Load the text data as the corpus"""
        _init_file_dir = os.path.dirname(__file__)
        if os.path.exists(os.path.join(_init_file_dir, "data/moby_dick.txt")):
            with open(os.path.join(_init_file_dir, "data/moby_dick.txt"), "r") as file:
                self.corpus = file.read()

        url = "https://lazyprogrammer.me/course_files/moby_dick.txt"
        response = requests.get(url)
        self.corpus = response.content.decode("utf-8")

    def __initialize_bigram_transition_probabilities(self):
        """ 
            Transition probabilities, represents the probability of going from 1 character to another.
            Initialize it to all 1s with a 26 x 26 matrix - since there are 26 letters in English
        """
        # initialize to a matrix with all 1s to handle add one smoothing
        self.M = np.ones((26, 26))

    def __initialize_unigram_distributions(self):
        """ 
            Probability distributions for the initial letters 
            Initialize it to an array of 26 - since there are 26 letters in English
        """
        self.pi = np.zeros(26)

    @staticmethod
    def letter_to_index(letter):
        """Use the binary conversion technique to convert alphabetical letters to int index"""
        return ord(letter) - 97

    def __update_transition_probability(self, char1, char2):
        """ Add 1 to the bigram transition probability matrix if we see a pattern from char1 to char2 """
        i = self.letter_to_index(char1)  # row index represents starting character
        j = self.letter_to_index(char2)  # column index represents ending character
        self.M[i, j] += 1

    def __update_unigram_distribution(self, char):
        i = self.letter_to_index(char)
        self.pi[i] += 1

    def build(self):
        """Build the language model for English Language, i.e the likelihood of letter combinations"""
        self.__load_data()
        self.__initialize_bigram_transition_probabilities()
        self.__initialize_unigram_distributions()
        # remove non-alphanumeric chars
        self.corpus = re.sub("[^a-zA-Z]", " ", self.corpus)

        # split into lower case tokens
        tokens = self.corpus.lower().split()

        # update the bigram transition probability and unigram probability
        for token in tokens:

            # update unigram probabilty
            self.__update_unigram_distribution(token[0])

            # update bigram probability
            for i in range(len(token) - 1):
                self.__update_transition_probability(token[i], token[i + 1])

        self.log_M = np.log(self.M / self.M.sum(axis=1, keepdims=True))
        self.log_pi = np.log(self.pi / self.pi.sum())

    def get_log_word_probability(self, word):
        """get the word log probability"""
        # take the first letter of the word to get the unigram probability
        first_letter_index = self.letter_to_index(word[0])
        log_unigram_prob = (
            self.log_pi[first_letter_index] if first_letter_index in np.arange(26) else 0
        )

        # get all the bigram probabilties for the rest of the characters
        log_bigram_prob = 0

        for i in range(len(word) - 1):
            starting_letter_index = self.letter_to_index(word[i])
            ending_letter_index = self.letter_to_index(word[i + 1])

            log_bigram_prob += (
                self.log_M[starting_letter_index][ending_letter_index]
                if (starting_letter_index in np.arange(26))
                and (ending_letter_index in np.arange(26))
                else 0
            )

        return log_unigram_prob + log_bigram_prob

    def get_sentence_log_probability(self, sentence):
        """get the sentence log probability"""
        # split the sentence into tokens
        tokens = sentence.split()
        # get the probability of a sentence
        log_sentence_prob = sum(
            [self.get_log_word_probability(token) for token in tokens]
        )
        return log_sentence_prob
