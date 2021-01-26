# [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) - Automated Speech and Sequence Generation

## Background (for interested readers)
This python project is a Markov Chain implementation which enables the user to generate speech (randomly) out of remembered training sets that can be sequences of any type whatsoever. Markov chains are used in a broad range in science. Escpecially in physics we use MCMC to simulate a system's state dynamically and to derive physical responses based on a mostly informational model and probabilities but also in machine learning MCMC is used either to generation, sorting and naive engines. Markov chains often come in handy as they are [ergodic](https://en.wikipedia.org/wiki/Ergodicity) (do not stuck in local minima that easy) thus quickly adapt to a specific usecase and may also work decently with limited amounts of data. A good example for this is the simulation of the [Ising model](https://en.wikipedia.org/wiki/Ising_model#Metropolis_algorithm) which uses a specific markov chain method called [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm). Such methods allow to sample directly according to complicated or unaccessible distributions i.e. common phrases have a higher likelihood to be sampled and so on. <br>
In addition to markov sampling, the training invokes [Bayesian classification](https://en.wikipedia.org/wiki/Bayes_classifier) for additional account of an a-priori likelihood by counting occurence frequencies of words regardless of the context and combining this with the likelihood that it may occur in a specific sequence aiming for a less biased estimate.

# Usage

## install
```
~/markov$ pip3 install .
```

## import
```python
from markov.chain import sequence
seq = sequence()
```
## Speech
Consider the following messages that occured in e.g. a chat room <br>
```python
msg_1 = "Great! I have to get back to work!"
msg_2 = "Sure, hearing from you!"
```
Simply train your chain
```python
seq.train(msg_1)
seq.train(msg_2)
```
and let this "bot" learn to speak like the chatting participants.

Teach the chain some Shakespeare

```python
sonnet = '''Weary with toil, I haste me to my bed,
The dear repose for limbs with travel tired;
But then begins a journey in my head,
To work my mind, when body’s work’s expired:
For then my thoughts (from far where I abide)
Intend a zealous pilgrimage to thee,
And keep my drooping eyelids open wide,
Looking on darkness which the blind do see:
Save that my soul’s imaginary sight
Presents thy shadow to my sightless view,
Which, like a jewel hung in ghastly night,
Makes black night beauteous and her old face new.
Lo, thus, by day my limbs, by night my mind,
For thee, and for myself, no quiet find.'''

seq.trainText(sonnet)

# generate text based on learned (from scratch)
print(seq.generate()) 

```
```
~$ but then begins a journey in my head to work my mind for thee and for myself no quiet find .
```
The output looks like some sentences were thrown together but actually the text is generated word by word based on sequential and bayesian probabilities.

For begun sentences like e.g. "I would" pass the words to ``generate()``

```python
begunSentence = ["I", "would"]
finalSentence = seq.generate(*begunSentence)
print(finalSentence)
```

<br>

## sequence completion
Lets have a closer look at general sequence and the ```next``` function.
Consider a winning/loosing streak of your favourite sports team
```python
priorGames = ['win', 'defeat', 'defeat', 'win', 'defeat', 'win', 'defeat', 'win', 'win']
seq.trainSeq(priorGames) 
```
one is obviously interested in the streak's continuation/completion. For this we infer ```next```. For performance and memory reasons, inputs are not saved which demands a further passing of the last two elements of your priorGames list (or if lazy the whole list) to ```next``` as the function needs to know the last two occurences to continue
```python
# seq.next(*priorGames[-2:]) == seq.next(*priorGames) => True
print(seq.next(*priorGames)) # use * to unpack the list into positional arguments
```
```
~$ []
```

an empty list is returned! This is where the program's architecture comes in: it cannot give a recommendation based on an unknown sequence as two wins never occured in a row! (see priorGames). This can be fixed with the ```improvise``` argument which fills such knowledge gaps with prior or more general estimations. This only triggers when there is no sequence probability

```python
# seq.next(*priorGames[-2:]) == seq.next(*priorGames) => True
print(seq.next(*priorGames, improvise=True)) # use * to unpack the list into positional arguments
```
```
~$ [('defeat', 0.4444444444444444), ('win', 0.5555555555555556)]
```
The returned array is always sorted by probability, the return shows that winning is more likely with a ~5% advantage or with 55.6%.
But what if we are interested in an arbitrary last sequence e.g. the last two games were defeats (the knwoledge still refers to priorGames)
```python
print(seq.next('defeat', 'win' improvise=True)) 
```
```
~$ [('.', 0.0), ('win', 0.18518518518518517), ('defeat', 0.2962962962962963)]
```
Note: improvise is not necessary anymore as this arrangement is known.
The '.' accounts for the possibility that this sequence ends here which is 0 as we used ```trainSeq``` if we would have used normal training for speech then this dot indicates the probability that the sentence terminates here. Also the probabilities do not add up to 1 anymore - this is due to the prior probabilities which were mixed in which breaks the normalization. Anyway the chain is sure that this will be a defeat!

<br>

Now with the understanding of ```next``` one could think of recursive calls to generate more than one state. This is ```generate``` - it calls ```next``` recursively.
So imagine we want to know how the next 5 games will turn out, then generate samples according to the probability of next.

```python
# call it more times
print(seq.generate(length=5)) 
print(seq.generate(length=5)) 
print(seq.generate(length=5)) 
```
```
~$ win defeat win defeat defeat
   win defeat defeat win win
   win defeat defeat win win
```

If a length is provided in sequence or speech generation this will be respected first, otherwise the length will be determined by probability. The upper three generated sequences are from the same distribution model but still random.
