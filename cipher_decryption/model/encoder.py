import string
import random
import numpy as np
import re


class Encoder:
    def __init__(self):
        self.__build_cipher_mapping()

    def __build_cipher_mapping(self):
        """ builds the subsitution cipher mapping for 26 letters """
        random.seed(7)
        letters_original = list(string.ascii_lowercase)
        letters_shuffled = random.sample(
            list(string.ascii_lowercase), len(list(string.ascii_lowercase))
        )
        self.encoder_cipher_mapping = {
            letters_original[i]: letters_shuffled[i] for i in range(len(letters_original))
        }

    def encode(self, message):
        """ method to encode the message using the cipher mapping """
        message = re.sub("[^a-zA-Z]", " ", message)
        message_tokens = list(message.lower())
        for i in range(len(message)):
            if message_tokens[i] in self.encoder_cipher_mapping:
                message_tokens[i] = self.encoder_cipher_mapping[message_tokens[i]]
        encoded_message = "".join(message_tokens)
        return encoded_message

    def decode(self, encoded_message, mapping):
        """ method to decode the message using the cipher mapping """
        message_tokens = list(encoded_message.lower())
        for i in range(len(encoded_message)):
            if message_tokens[i] in mapping:
                message_tokens[i] = mapping[message_tokens[i]]
        decoded_message = "".join(message_tokens)
        return decoded_message
