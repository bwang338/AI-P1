import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line not in vocab:
                line = None
            if not line in bow.keys():
                bow[line] = 1
            else:
                num = bow.get(line)
                num = num + 1
                bow[line] = num
    return bow

#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    totalFiles = len(training_data)
    for label in label_list:
        rightLabel = 0
        logprob[label] = 0
        for data in training_data:
            if data['label'] == label:
                rightLabel = rightLabel + 1
        prob = (rightLabel + 1)/(totalFiles + 2)
        logprob[label] = math.log(prob)
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here
    for word in vocab:
        word_prob[word] = 0
    word_prob[None] = 0
    totalWC = 0
    vocabSize = len(vocab)
    for data in training_data:
        if data['label'] == label:
            for word in data['bow'].keys():
                if word not in vocab:
                    word_prob[None] += data['bow'][None]
                    totalWC += data['bow'][None]
                else:
                    word_prob[word] += data['bow'][word]
                    totalWC += data['bow'][word]
    for key in word_prob.keys():
        word_prob[key] = (word_prob[key] + 1)/(totalWC + vocabSize + 1)
        word_prob[key] = math.log(word_prob[key])
    return word_prob


##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(data, ['2020', '2016'])
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, data, '2020')
    
    return retval

#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    wordFreq = create_bow(model['vocabulary'], filepath)
    prob2016 = 0
    prob2020 = 0
    for key, value in wordFreq.items():
        prob2016 += model['log p(w|y=2016)'][key] * value
        prob2020 += model['log p(w|y=2020)'][key] * value
    prior2016 = model['log prior']['2016']
    prior2020 = model['log prior']['2020']
    return2016 = prob2016 + prior2016
    return2020 = prob2020 + prior2020
    if max(return2016, return2020) == return2016:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'
    retval['log p(y=2016|x)'] = return2016
    retval['log p(y=2020|x)'] = return2020
    return retval
