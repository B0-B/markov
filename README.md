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
and let this "bot" learn to speak like the participants.

We can also let him learn Shakespeare
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
The output looks like he has thrown some sentences together but actually the text is generated word by word based on sequential and bayesian probabilities.

<br>

## sequence completion
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
~$ 
```