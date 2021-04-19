from __future__ import print_function

import numpy as bb8
import handle_image as hi
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import image_operations as iop


def prepare_data_for_network(letters):
    data_for_network = []
    for letter in letters:
        scaled = iop.scale_image(letter)
        data_for_network.append(iop.image_to_vector(scaled))

    return data_for_network


def convert_output():
    alphabet = create_alphabet()
    outputs = []
    for i in range(len(alphabet)):
        output = bb8.zeros(len(alphabet))
        output[i] = 1
        outputs.append(output)
    return bb8.array(outputs)


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, k_means):
    alphabet = create_alphabet()
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result


def do_fuzzywuzzy_stuff(vocabulary, extracted_text):
    new_string = ""
    for extracted_word in list(extracted_text.split(' ')):
        # print(extracted_word)
        found_word = find_closest_word(vocabulary, extracted_word)
        # print(found_word)
        if found_word is None:
            new_string += extracted_word + ""
        else:
            new_string += found_word + " "
        new_string.rstrip()
    return new_string


def find_closest_word(vocabulary, extracted_word):
    list_of_words = list(vocabulary.keys())
    closest_word =""
    closest_words = []
    lowest_distance=100000
    for word in list_of_words:
        if word == extracted_word:
            return word
        else:
            distance = fuzz.ratio(word,extracted_word)
            if distance < lowest_distance:
                lowest_distance = distance
                closest_word = word
            elif distance == lowest_distance:
                closest_words.append([word, distance, vocabulary[word]])

    highest_occurance = 0
    final_word = ""
    for word_distance_occur in closest_words:
        if int(word_distance_occur[2])>highest_occurance:
            final_word=word_distance_occur[0]
            highest_occurance=int(word_distance_occur[2])

    return final_word

# code for levenshein taken from https://www.datacamp.com/community/tutorials/fuzzy-string-python

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = bb8.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return distance[row][col]

def create_alphabet():
    alphabet_upper = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                      'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž']
    alphabet = []
    for char in alphabet_upper:
        alphabet.append(char)

    for char in alphabet_upper:
        alphabet.append(char.lower())
    return alphabet

