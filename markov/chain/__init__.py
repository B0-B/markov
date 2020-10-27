#!/usr/bin/env python3
import numpy as np 

class sequence:

    '''
    Sequential Markov chain implementation for e.g.
    a text or bit generation/completion/suggestion. This algorithm
    remembers/learns sequences from training material
    like a text messages from a chat, sequence of wins/lose
    sequences in sport results and so on. It can then be 
    used to either generate random new sequences according 
    to the probability distributions from scratch, or just 
    predict new states based on a sequence at hand. Note:
    the chain can only process objects which it has seen.
    '''

    # example filling method
    db_template = {
        "words": {
            "weight": 0,    
            "": {
                "weight": 1,
                "yo": {
                    ".": {"weight": 0},
                    "weight": 1,
                    "was": {
                        ".": {"weight": 0},
                        "weight": 1,
                        "geht": {"weight": 1}
                    }
                }
            },
            "ich": {

                "weight": 1,
                ".": {"weight": 0},
                "weiss": {
                    
                    "weight": 1,
                    ".": {"weight": 0},
                    "schon": {"weight": 1},
                    "nicht": {"weight": 1}
                }
            }
        },

        "dictionary": {},

        # remember the number of messages
        "messages": 0,

        # remember the mean length per message
        "meanLength": 4,
    }
    db = dict(db_template) # use a template copy


    # object variables
    punctuations = '''"'!?.,;+/:()'''

    def generate(self, *sequence, length=None, gapFill=False):

        # decide on provided sequence

        if len(sequence) == 0:

            # sample the first word
            sentence = [self.sample(self.next(""))]

            # second word
            sentence.append(self.sample(self.next("", sentence[-1])))
        
        elif len(sequence) == 1:

            # first
            sentence = []
            if sequence[0] in self.db["words"][""]:
                sentence.append(self.sample(self.next("", sequence[0])))
            else:
                sentence.append(self.sample( list(self.db["dictionary"].items()) ))

            # second
            sentence.append(self.next("", sentence[-1]))
        
        else:

            sentence = list(sequence)


        # get new words by markov chain procedure
        while True:

            if length == None:

                new = self.sample(self.next(sentence[-2], sentence[-1], improvise=gapFill))

                # decide when to break the sentence
                if new == "." or new == "":
                    sentence.append(".")
                    break
                else:
                    # check for the length
                    if gapFill and len(sentence) > 2*self.db["meanLength"]:
                        u = np.random.uniform(0, 1)
                        if u < 0.3:
                            sentence.append(".")
                            break
                    else:
                        sentence.append(new)
            
            else:
                new = self.sample(self.next(sentence[-2], sentence[-1], improvise=True))

                if new not in ["", "."]:
                    sentence.append(new)
                if len(sentence) == length:
                    break

        return ' '.join(sentence)

    def next(self, *lastTwoWords, improvise=False):

        '''
        Returns a prediction of words sorted by their probability to occur.
        lastTwoWords: either a two word list args 'Hello', 'World' or a whole 
        msg list but note that only the last two words of a message will be
        processed. If you provide a list mark it with a "*".
        '''

        if len(lastTwoWords) == 1:

            if lastTwoWords[0] != "": # check that the provided predecessor is empty i.e. the beginning of the sentence

                raise ValueError(f'if only one word is provided it must be "" and not {lastTwoWords[0]} ...')

            # get all words and possibilities continuing from this node
            unsortedTuples = list(self.db["words"][""].items())

            # get the partition size
            N = self.db["words"][""]["weight"]

        else:
            #print('check', self.db["words"]['win']['win'])
            # check if the words are known to the data base
            if lastTwoWords[-2] not in self.db["words"]:
                unsortedTuples = []
            else:
                if lastTwoWords[-1] not in self.db["words"][lastTwoWords[-2]]:
                    unsortedTuples = []
                else:
                    # if known do
                    unsortedTuples = list(self.db["words"][lastTwoWords[-2]][lastTwoWords[-1]].items())

                    # get the partition size
                    N = self.db["words"][lastTwoWords[-2]][lastTwoWords[-1]]["weight"]

        # apply priors to the unsorted tuple set
        unsortedTuplesWithPriors = []
        for i in range(len(unsortedTuples)):
            tuple = unsortedTuples[i]
            if tuple[0] != 'weight': # exclude the weight as it is no word
                unsortedTuplesWithPriors.append( (tuple[0], tuple[1]["weight"]/N * self.prior(tuple[0])) )

        # imporvise when no sequence matches
        if improvise and len(unsortedTuplesWithPriors) == 0:

            unsortedTuplesWithPriors = list(self.db['dictionary'].items())
            # normalize
            for i in range(len(unsortedTuplesWithPriors)):
                unsortedTuplesWithPriors[i] = (unsortedTuplesWithPriors[i][0], self.prior(unsortedTuplesWithPriors[i][0]))

        # sort tuples
        sortedTuplesWithPriors = self.sort(unsortedTuplesWithPriors)
        
        return sortedTuplesWithPriors
  
    def prior(self, word):

        '''
        Dictionary look up - computes and returns the prior probability i.e.
        the likelihood for a word to occur in general.
        '''

        if word not in self.db['dictionary']:
            self.db['dictionary'][word] = 0 # just list the word
        
        return self.db['dictionary'][word]/float(self.db["words"]["weight"])

    def sample(self, wordTuples):

        '''
        This function samples according to the distribution
        given by the prob. in the wordTuples using classic
        inversion sampling.
        '''

        if wordTuples == []:
            return ""

        cumulative = []
        for i in range(len(wordTuples)):
            if cumulative == []:
                cumulative.append(wordTuples[i][1])
            else:
                cumulative.append(cumulative[-1] + wordTuples[i][1])
        
        # sample in the allowed range e.g. 0 to 0.39 since the
        # probabilities in the wordTuples are not normalized
        u = np.random.uniform(0, cumulative[-1])

        # get the inverse of the probability distribution
        for i in range(len(cumulative)):
            if u <= cumulative[i]:
                return wordTuples[i][0]

    def sort(self, wordTuples):

        '''
        wordTuples is a list of tuples [word, probability] = [str, float]
        '''

        sorted = []
        for k, v in wordTuples:
            if len(sorted) < 1:
                sorted.append((k, v))
            elif len(sorted) < 2:
                if v > sorted[0][1]:
                    sorted.insert(1, (k, v))
                else:
                    sorted.insert(0, (k, v))
            else:
                slippedIn = False
                for i in range(len(sorted)-1):
                    if v <= sorted[i+1][1] and v > sorted[i][1]:
                        sorted.insert(i+1, (k, v))
                        slippedIn = True
                        break
                if not slippedIn:
                    if v <= sorted[0][1]:
                        sorted.insert(0, (k, v))
                    else:
                        sorted.append((k, v))

        return sorted

    def train(self, message=str(), endings=True):

        # preprocess the message
        for p in self.punctuations: # remove punctuations
            if p in message:
                message = ''.join(message.split(p))
        message = ' '.join(message.split('\n')) # remove linebreaks
        message = message.split(' ')

        # iterate through message words
        msg = [] # this will be a normed massage format which will be saved
        for i in range(len(message)):

            # lower case only for matching
            w = message[i].lower()

            # ---- DATABASE updating ---- #
            # check if w is the first word in sentence
            if len(msg) == 0:

                # update 1st level
                self.db["words"][""]["weight"] = self.db["words"][""]["weight"] + 1

                # 2nd
                if w in self.db["words"][""]:
                    self.db["words"][""][w]["weight"] = self.db["words"][""][w]["weight"] + 1
                else:
                    self.db["words"][""][w] = {"weight": 1, ".": {"weight": 0}}
                
                # if endings are enabled
                if endings:
                    if len(msg) == i+1: # the first word is the  only word
                        self.db["words"][""][w]["."]["weight"] = self.db["words"][""][w]["."]["weight"] + 1

            elif len(msg) == 1:

                # update 3rd level
                if w in self.db["words"][""][msg[-1]]:
                    self.db["words"][""][msg[-1]][w]["weight"] = self.db["words"][""][msg[-1]][w]["weight"] + 1
                else:
                    self.db["words"][""][msg[-1]][w] = {"weight": 1, ".": {"weight": 0}}

                # if endings are enabled
                if endings:
                    if len(msg) == i+1: # last word
                        self.db["words"][msg[-1]][w]["."]["weight"] = self.db["words"][msg[-1]][w]["."]["weight"] + 1

            else:

                # check 1st level
                if msg[-2] in self.db["words"]:
                    self.db["words"][msg[-2]]["weight"] = self.db["words"][msg[-2]]["weight"] + 1
                else:
                    self.db["words"][msg[-2]] = {"weight": 1, ".": {"weight": 0}}
                
                # 2nd
                if msg[-1] in self.db["words"][msg[-2]]:
                    self.db["words"][msg[-2]][msg[-1]]["weight"] = self.db["words"][msg[-2]][msg[-1]]["weight"] + 1
                else:
                    self.db["words"][msg[-2]][msg[-1]] = {"weight": 1, ".": {"weight": 0}}

                # 3rd
                if w in self.db["words"][msg[-2]][msg[-1]]:
                    self.db["words"][msg[-2]][msg[-1]][w]["weight"] = self.db["words"][msg[-2]][msg[-1]][w]["weight"] + 1
                else:
                    self.db["words"][msg[-2]][msg[-1]][w] = {"weight": 1}

                # if endings are enabled
                if endings:
                    if len(msg) == i+1: # last word
                        self.db["words"][msg[-1]][w]["."]["weight"] = self.db["words"][msg[-1]][w]["."]["weight"] + 1

            # add word to msg
            msg.append(w)

            # add word to dictionary
            if w not in self.db['dictionary']:
                self.db['dictionary'][w] = 1
            else:
                self.db['dictionary'][w] = self.db['dictionary'][w] + 1
            # ---------------------------- #
            
        # count words
        self.db["words"]["weight"] = self.db["words"]["weight"] + len(msg)

        # count the message
        self.db['messages'] = self.db['messages'] + 1

        # update mean message length from persistent mean
        self.db['meanLength'] = (self.db['meanLength'] + len(msg)/float(self.db['messages'])) * self.db['messages']/(self.db['messages'] + 1)

    def trainSeq(self, sequence):
        '''
        Sequence must be a list object.
        '''
        self.train(' '.join(sequence), endings=False)

    def trainText(self, text):

        messages = []
        # split the text into messages=sentences
        for p in ['!', '?', '...', '..', ';']: # exchange punctuations with
            text = '.'.join(text.split(p))
        messages = text.split('.')
        for message in messages:
            self.train(message)