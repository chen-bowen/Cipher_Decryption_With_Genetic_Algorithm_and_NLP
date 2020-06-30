import numpy as np
import random
import string
from model.language_model import LanguageModel
from model.encoder import Encoder
from copy import copy


class GeneticAlgorithm:

    POOL_SIZE = 20
    OFFSPRING_POOL_SIZE = 5
    NUM_ITER = 250

    def __init__(self):
        self.__initialize_dna_pool()
        self.__initialize_encodeder()
        self.__initialize_language_model()

    def __initialize_dna_pool(self):
        """ Generates a DNA pool """
        # get all 26 letters
        self.all_letters = list(string.ascii_lowercase)
        # generate 20 random permutation of the all_letters as the dna_pool
        self.dna_pool = [
            "".join(list(np.random.permutation(self.all_letters)))
            for _ in range(self.POOL_SIZE)
        ]
        self.offspring_pool = []

    def __initialize_encodeder(self):
        """ Initialize the Encoder object as a class attribute encoder """
        self.encoder = Encoder()

    def __initialize_language_model(self):
        """ Initialize the LanguageModel object as a class attribute m """
        self.m = LanguageModel()

    @staticmethod
    def random_swap(sequence):
        """ random swap two characters at random positions """
        # generate random positions
        index_1, index_2 = random.sample(list(np.arange(len(sequence))), 2)
        # swap positions
        sequence_copy = copy(sequence)
        seq_list = list(sequence_copy)
        seq_list[index_1], seq_list[index_2] = seq_list[index_2], seq_list[index_1]
        return "".join(seq_list)

    def evolve_offspring(self):
        """ evolves offspring by random swaps for every dna sequence in the dna pool """
        # import pdb

        # pdb.set_trace()
        for dna in self.dna_pool:
            # evolve 10 offsprings for each dna in the dna pool
            self.offspring_pool += [
                self.random_swap(dna) for _ in range(self.OFFSPRING_POOL_SIZE)
            ]
        return self.offspring_pool + self.dna_pool

    def train(self, initial_message):
        """ Train the Genetic Algorithm by the real data from inital_message """
        # initialize vars
        self.avg_scores_per_iter = np.zeros(self.NUM_ITER)
        self.best_scores_per_iter = np.zeros(self.NUM_ITER)
        self.best_dna = None
        self.best_mapping = None
        self.best_score = float("-inf")
        dna_scores = {}

        # get encoded message from initial message
        encoded_message = self.encoder.encode(initial_message)

        # iterate
        for i in range(self.NUM_ITER):
            # only evolve offspring on the second iteration onwards
            if i > 0:
                self.dna_pool = self.evolve_offspring()

            # find scores for each dna
            for dna in self.dna_pool:
                # get current mapping from current dna
                curr_mapping = {
                    original_letter: encoded_letter
                    for original_letter, encoded_letter in zip(self.all_letters, dna)
                }
                # decode using current mapping and get the score
                curr_decoded_message = self.encoder.decode(encoded_message, curr_mapping)
                dna_scores[dna] = self.m.get_sentence_log_probability(
                    curr_decoded_message
                )

                # record the best performing dna sequence
                if dna_scores[dna] > self.best_score:
                    self.best_score = dna_scores[dna]
                    self.best_mapping = curr_mapping
                    self.best_dna = dna

            # get the current generation average scores
            self.avg_scores_per_iter[i] = np.mean(list(dna_scores.values()))
            self.best_scores_per_iter[i] = self.best_score

            # choose the top 5 best performing dna sequece to pass on to the next generation
            sorted_dna_curr_gen = sorted(
                dna_scores.items(), key=lambda x: x[1], reverse=True
            )
            self.dna_pool = [sequence[0] for sequence in sorted_dna_curr_gen[:5]]

            if i in np.arange(0, 251, 50):
                print(
                    "\n iter: {},".format(i),
                    "log likelihood: {},".format(self.avg_scores_per_iter[i]),
                    "best likelihood so far: {}".format(self.best_score),
                    "\n decoded_message: \n {} \n".format(
                        self.encoder.decode(encoded_message, self.best_mapping)
                    ),
                )
