# [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) - Automated Speech and Sequence Generation

## Background (for interested readers)
This python project is a Markov Chain implementation which enables the user to generate speech (randomly) out of remembered training sets that can be sequences of any type whatsoever. Markov chains are used in a broad range in science. Escpecially in particle physics and quantum systems we use MCMC to simulate this system state dynamically and to derive physical responses based on a mostly informational model and probabilities. Markov chains often come in handy as they quickly adapt to a specific training and may also work with limited amounts of data. A good example for this is the simulation of the [Ising model](https://en.wikipedia.org/wiki/Ising_model#Metropolis_algorithm) which uses a specific markov chain method called [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) which specificly uses (physical) Boltzmann weights. Such methods allow to sample directly according to complicated distributions i.e. common phrases have a higher likelihood to occur in a sentence and so on. <br>
In this project a very similar approach is used but in a simpler manner. Instead of simulating/generating new physical states out of a state sequence one may generate new words based on some previous word sequence (for instance a started sentence). In addition to the sampling with markov, the training works with [Bayesian classification](https://en.wikipedia.org/wiki/Bayes_classifier) moreover Bayes is invoked for additional account of an a-priori distribution by counting how frequent words occur in general and combining this with the likelihood that it may occur in a specific sequence. This Bayesian prior is then also used during the sampling process by joining both probabilities (prior and sequential) and should lead to a "less biased" estimation/recommendation.

# Usage
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

We can also teach the chain some Shakespeare

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

<br>

## sequence completion
We want to have a look at general sequences which is straight forward and we want to have a closer look into the ```next``` function.
Consider a winning/loosing streak of a sports team
```python
priorGames = ['win', 'defeat', 'defeat', 'win', 'defeat', 'win', 'defeat', 'win', 'win']
seq.trainSeq(priorGames) 
```
One may be interested how this streak will continue or be 'completed'. For this we infer ```next```. Since the chain does not save your inputs for performance and memory reasons just pass the last two elements of your priorGames list or if lazy the whole list to next() as the function needs to know the last two occurences to continue
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