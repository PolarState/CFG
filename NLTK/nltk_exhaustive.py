import nltk
from nltk import CFG
from nltk.parse.generate import generate

# 1. Define a Context-Free Grammar (CFG)
# By convention, the start symbol is 'S'.
grammar = CFG.fromstring("""
  S -> NP VP
  NP -> Det N | 'I'
  VP -> V NP | V
  Det -> 'the' | 'a'
  N -> 'man' | 'dog' | 'food'
  V -> 'saw' | 'ate'
""")

cfg3b_str = """
    22 -> 21 20 | 20 19
    
    21 -> 18 16 | 16 18 17
    20 -> 17 16 18 | 16 17
    19 -> 16 17 18 | 17 18 16
    
    18 -> 15 14 13 | 14 13
    17 -> 14 13 15 | 15 13 14
    16 -> 15 13 | 13 15 14
    
    15 -> 12 11 10 | 11 12 10
    14 -> 11 10 12 | 10 11 12
    13 -> 11 12 | 12 11
    
    12 -> 8 9 7 | 9 7 8
    11 -> 8 7 9 | 7 8 9
    10 -> 7 9 8 | 9 8 7
    
    9 -> '321' | '21'
    8 -> '32' | '312'
    7 -> '31' | '123'
"""

cfg3b_short_str = """
    16 -> 15 13
    15 -> 12 11 10 | 11 12 10
    14 -> 11 10 12 | 10 11 12
    13 -> 11 12 | 12 11
    12 -> 8 9 7 | 9 7 8
    11 -> 8 7 9 | 7 8 9
    10 -> 7 9 8 | 9 8 7
    9 -> '321' | '21'
    8 -> '32' | '312'
    7 -> '31' | '123'
"""

# Sum(Downstream NT options ^ number of NT tokens per rule) per NT generation rule
# 12 -> 8 9 7 | 9 7 8
# 9 -> ''3''2''1'' | '2''1'
# 8 -> '3''2' | '3''1''2'
# 7 -> '3''1' | '1''2''3'
#
# 9, 8, & 7 each have two generation rules. So it's 2 ^ the number of NT symbols in each sequence
# 12 has two genration rules with a length of 3, so 2 ^ 3 for each side or 2 ^ 3 + 2 ^ 3
# 
# For the slightly longer sequence, the pattern continues, we keep raising the number of options
# possible to the length of the sequence, for each sequence possible, then sum them.
# 13 -> 11 12 | 12 11
# 11 -> 8 7 9 | 7 8 9
# 12 -> 8 9 7 | 9 7 8
# 9 -> '3''2''1' | '2''1'
# 8 -> '3''2' | '3''1''2'
# 7 -> '3''1' | '1''2''3'
# Can be recursively reframed as:
# 13 -> 11 12 | 12 11
# 11 -> 2 ^ 3 + 2 ^ 3 possibilities
# 12 -> 2 ^ 3 + 2 ^ 3 possibilities
# So since each NT sequence of 13 is of length two and there are two of them, we get:
# (2 ^ 3 + 2 ^ 3) ^ 2 + (2 ^ 3 + 2 ^ 3) ^ 2


# 22 -> (81,149,446,665,341,947,263,502,314,897,408 ^ 2) + 162,298,893,328,322,134,789,633,131,675,648 * 81,149,446,665,341,947,263,502,314,897,408
# 21 -> (34,363,932,672 ^ 2) + 34,363,932,672 ^ 2 * 68,719,476,736 = 81,149,446,665,341,947,263,502,314,897,408
# 20 -> 68,719,476,736 * (34,363,932,672 ^ 2) + 34,363,932,672 * 68,719,476,736 = 81,149,446,665,341,947,263,502,314,897,408
# 19 -> (34,363,932,672 ^ 2) * 68,719,476,736 + 68,719,476,736 * 34,363,932,672 ^ 2 = 162,298,893,328,322,134,789,633,131,675,648
# 18 -> (8,192 ^ 2) * 512 + 8,192 * 512 = 34,363,932,672
# 17 -> (8,192 ^ 2) * 512 + 8,192 * 512 * 8,192 = 68,719,476,736
# 16 -> 8,192 * 512 + 512 * 8,192 * 8,192 = 34,363,932,672
# 15 -> 16 ^ 3 + 16 ^ 3 = 8,192
# 14 -> 16 ^ 3 + 16 ^ 3 = 8,192
# 13 -> 16 ^ 2 + 16 ^ 2 = 512
# 12 -> 2 ^ 3 + 2 ^ 3 = 16
# 11 -> 2 ^ 3 + 2 ^ 3 = 16
# 10 -> 2 ^ 3 + 2 ^ 3 = 16
# 9 -> 2
# 8 -> 2
# 7 -> 2

# As I watch the generations steam by, it reminds me that there is definite and finite structure here.
# What is the limit of the structure?
# In one dimension there are there words. There are actually relatively few words just six in total:
# '321', '21', '32', '312', '31', '123'. 



cfg3b = CFG.fromstring(cfg3b_short_str)


# 2. Enumerate the sentences
# The generate function returns an iterator of tokenized sentences.
# with open('output.txt', "w") as f:
#     for sentence in generate(cfg3b):
#         # Join the list of tokens into a single string for readability
#         f.write(''.join(sentence))


num_productions = 0
for sentence in generate(cfg3b, ):
    # Join the list of tokens into a single string for readability
    # print(''.join(sentence))
    num_productions += 1

print(num_productions)
# cfg3b.check_coverage('2112331212331221')




(34,363,932,672 ^ 2) * 68,719,476,736

#       1 1   1 1
#    34,363,932,672
#  *              2
#  ----------------
#    68,727,865,344

#           214 1  
#    34,363,932,672
#  *             70
#  ----------------
#            56,340