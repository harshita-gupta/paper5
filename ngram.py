from nltk import ngrams
marquez = 'books/One Hundred Years of Solitude.txt'
with open(marquez, 'r') as myfile:
    book = myfile.read().replace('\n', ' ').split()
uniquewords = set(book)


sentence = 'this is a foo bar sentences and i want to ngramize it'
n = 6
sixgrams = ngrams(sentence.split(), n)
# for grams in sixgrams:
#     print grams
print sixgrams
