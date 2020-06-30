from model.genetic_algorithm import GeneticAlgorithm
from model.encoder import Encoder

original_message = """I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
"""

g = GeneticAlgorithm()
g.train(original_message)

best_mapping = g.best_mapping

encoder = Encoder()
encoded_message = encoder.encode(original_message)
decoded_message = encoder.decode(encoded_message, best_mapping)
