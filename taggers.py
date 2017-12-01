#!/usr/bin/env python3
import sys
import math
import numpy

# read files
train = open(sys.argv[1]).read().strip()
test = open(sys.argv[2]).read().strip()
train = train.split('\n')
test = test.split('\n')
# data collection
vocabulary_size = len(train) - 1
tag_dictionary = {} # store all constituent tags a word can have
count_dictionary_word = {} # store counts of word unigrams and bigrams (wordi, tagi) 
count_dictionary_tag = {} # store counts of tag unigrams and bigrams (ti-1, ti)
backpointer_list = [] # list of backpointers
alpha_table = {}
beta_table = {}
mu_table = {} # a dict of lists storing path probs
backpointer_table = {} # backpointer table
posterior_probability = {} # tracking accumulating probabilities in forward-backward algorithm
one_count_tag_bigram = {}
one_count_word_tag = {}
novel_word = 'OOV'
tag_dictionary[novel_word] = []
count_dictionary_word[novel_word] = 0

# read training set
def read_file(train_file):
    if len(train_file) < 1:
        return None
    # processing the first line
    first_line = train_file[0]
    first_word, first_tag = first_line.split('/')[0], first_line.split('/')[1]
    if first_tag not in one_count_word_tag:
        one_count_word_tag[first_tag] = 0
    if first_line not in count_dictionary_word:
        count_dictionary_word[first_line] = 1
        one_count_word_tag[first_tag] += 1
    else:
        count_dictionary_word[first_line] += 1   

    for idx in range(1, len(train_file)):
        line = train_file[idx]
        word, tag = line.split('/')[0], line.split('/')[1]
        if tag != '###':
            if tag not in tag_dictionary[novel_word]:
                tag_dictionary[novel_word].append(tag) # collecting tags of all novel words
        # store all tags of a word
        if word not in tag_dictionary:
            tag_dictionary[sys.intern(word)] = [tag]
        else:
            if tag not in tag_dictionary[word]:
                tag_dictionary[sys.intern(word)].append(tag)
        # store counts of unigrams and bigrams
        if word not in count_dictionary_word:
            count_dictionary_word[sys.intern(word)] = 1
        else:
            count_dictionary_word[sys.intern(word)] += 1
        ######
        if tag not in one_count_word_tag:
            one_count_word_tag[tag] = 0
        ######
        if tag not in count_dictionary_tag:
            count_dictionary_tag[sys.intern(tag)] = 1

        else:
            count_dictionary_tag[sys.intern(tag)] += 1
        if line not in count_dictionary_word:
            count_dictionary_word[sys.intern(line)] = 1
            one_count_word_tag[tag] += 1
        else:
            count_dictionary_word[sys.intern(line)] += 1
            if count_dictionary_word[line] == 2:
                one_count_word_tag[tag] -= 1
        previous_tag = train_file[idx - 1].strip().split('/')[1]
        ###
        if previous_tag not in one_count_tag_bigram:
            one_count_tag_bigram[previous_tag] = 0
        ###
        tag_bigram = sys.intern(previous_tag + '_' + tag)
        if tag_bigram not in count_dictionary_tag:
            count_dictionary_tag[tag_bigram] = 1
            one_count_tag_bigram[previous_tag] += 1 # add one to number of singleton
        else:
            count_dictionary_tag[tag_bigram] += 1
            if count_dictionary_tag[tag_bigram] == 2:
                one_count_tag_bigram[previous_tag] -= 1 # for tag bigrams, reduce number of singleton by 1
    return None

# helper functions for prob
def prob(prev_tag, tag, word):
    word_count = len(tag_dictionary)
    value = 0
    lambda_pwt = 0 
    lambda_ptt = 0 # initialize the lambda value for one count smoothing
    p_tw = 0
    tag_bigram = prev_tag + '_' + tag
    tag_bigram_occurrence = 0
    word_tag = word + '/' + tag
    word_tag_occurrence = 0
    if word == '###':
        p_tw = math.log(1)
    else:
        if word_tag not in count_dictionary_word:
            word_tag_occurrence = 0
        else:
            word_tag_occurrence = count_dictionary_word[word_tag]
        lambda_pwt = one_count_word_tag[tag] + 1
        backoff_word_tag = float(count_dictionary_word[word] + 1) * 1.0 / (vocabulary_size + word_count)
        p_tw = math.log(((word_tag_occurrence + lambda_pwt * backoff_word_tag) * 1.0) / (count_dictionary_tag[tag] + lambda_pwt))
    if tag_bigram not in count_dictionary_tag:
        tag_bigram_occurrence = 0
    else:
        tag_bigram_occurrence = count_dictionary_tag[tag_bigram]
    lambda_ptt = one_count_tag_bigram[prev_tag] + 1
    backoff_tag_bigram = float(count_dictionary_tag[tag]) * 1.0 / vocabulary_size
    p_tt = math.log(((tag_bigram_occurrence + lambda_ptt * backoff_tag_bigram)* 1.0) / (count_dictionary_tag[prev_tag] + lambda_ptt))
    value = p_tt + p_tw
    return value

# Viterbi Tagging
def viterbi_tagging():
    mu_table['###'] = {}
    mu_table['###'][0] = math.log(1)
    # backpointer initialized as None. When traceback encounters None, cease action
    for idx in range(1, len(test)):
        line = test[idx]
        word = line.split('/')[0]
        if word not in tag_dictionary:
            word = novel_word
        prev_word = test[idx - 1].split('/')[0]
        if prev_word not in tag_dictionary:
            prev_word = novel_word
        #best_mu = -float("inf")
        for tag in tag_dictionary[word]:
            for prev_tag in tag_dictionary[prev_word]:
                p = prob(prev_tag, tag, word) # calculate arch probability
                mu = mu_table[prev_tag][idx - 1] + p
                if tag not in mu_table:
                    mu_table[tag] = {}
                if idx not in mu_table[tag]:
                    mu_table[tag][idx] = -float('inf')
                if tag not in backpointer_table:
                    backpointer_table[tag] = {}
                #print("mu is: " + str(mu))
                if mu > mu_table[tag][idx]: # or >=?
                    mu_table[tag][idx] = mu
                    backpointer_table[tag][idx] = prev_tag
                    # backpointer = prev_tag
                    # backpointer = prev_word + '/' + prev_tag
        # backpointer_list.append(backpointer)
    traceback = {}
    tag_n = '###'
    traceback[len(test) - 1] = tag_n
    for idx in range(len(test) - 1, 0, -1):
        traceback[idx - 1] = backpointer_table[traceback[idx]][idx]
    # backpointers = backpointer_list[::-1]
    return traceback

def forward_backward_tagging():
    alpha_table['###'] = {}
    alpha_table['###'][0] = math.log(1)
    for idx in range(1, len(test)):
        word = test[idx].split('/')[0]
        prev_word = test[idx - 1].split('/')[0]
        if word not in tag_dictionary:
            word = novel_word
        if prev_word not in tag_dictionary:
            prev_word = novel_word
        for tag in tag_dictionary[word]:
            for prev_tag in tag_dictionary[prev_word]:
                p = prob(prev_tag, tag, word)
                if tag not in alpha_table:
                    alpha_table[tag] = {}
                if idx not in alpha_table[tag]:
                    alpha_table[tag][idx] = float("-inf")
                alpha_table[tag][idx] = numpy.logaddexp(alpha_table[tag][idx], alpha_table[prev_tag][idx - 1] + p)
    Z = alpha_table['###'][len(test) - 1]
    beta_table['###'] = {}
    beta_table['###'][len(test) - 1] = math.log(1)
    for idx in range(len(test) - 1, 0, -1):
        word_backward = test[idx][0]
        prev_word_backward = test[idx - 1][0]
        if word_backward not in tag_dictionary:
            word_backward = novel_word
        if prev_word_backward not in tag_dictionary:
            prev_word_backward = novel_word
        for tag_backward in tag_dictionary[word_backward]:
            if idx not in posterior_probability:
                print(tag_backward)
                print(beta_table)
                print(beta_table[tag_backward][idx])
                print(alpha_table[tag_backward][idx])
                combined_prob = alpha_table[tag_backward][idx] + beta_table[tag_backward][idx]
                slot = [tag_backward, combined_prob]
                posterior_probability[idx] = []
                posterior_probability[idx] = slot
                #posterior_probability[idx] = tuple(tag_backward, alpha_table[tag_backward][idx] + beta_table[tag_backward][idx])
            else:
                # update posterior probability only when the sum of alpha and beta exceeds the last posterior probability
                if (alpha_table[tag_backward][idx] + beta_table[tag_backward][idx]) > posterior_probability[idx][1]:
                    # using tuple to store the tag and value of alpha and beta probs
                    posterior_probability[idx] = tuple(tag_backward, alpha_table[tag_backward][idx] + beta_table[tag_backward][idx])
            for prev_tag_backward in tag_dictionary[prev_word_backward]:
                p = prob(prev_tag_backward, tag_backward, word_backward)
                if prev_tag_backward not in beta_table:
                    beta_table[prev_tag_backward] = {}
                prev_idx = idx - 1
                if prev_idx not in beta_table[prev_tag_backward]:
                    beta[prev_tag_backward][prev_idx] = float("-inf") # if prev index not in the beta table, initialize it.
                beta_table[prev_tag_backward][prev_idx] = numpy.logaddexp(beta_table[prev_tag_backward][prev_idx], beta[tag_backward][idx] + p)
    traceback = {}
    start_pos = 0
    traceback[start_pos] = '###'
    for idx in posterior_probability:
        traceback[idx] = posterior_probability[idx][start_pos]
    return traceback

def compute_perplexity():
    perplexity = 0.0
    for idx in range(1, len(test)):
        word, tag = test[idx].split('/')[0], test[idx].split('/')[1]
        if word not in tag_dictionary:
            word = novel_word
        prev_tag = test[idx - 1].split('/')[1]
        p = prob(prev_tag, tag, word)
        perplexity += p
    perplexity_per_word = math.exp(-1 * perplexity / (len(test) - 1))
    print("Model perplexity per tagged test word: {:.3f}".format(perplexity_per_word))
    return None

def compute_accuracy(backpointers, isPosterior):
    correct_test_tokens = 0
    wrong_test_tokens = 0
    known_correct_tokens = 0
    known_wrong_tokens = 0
    novel_correct_tokens = 0
    novel_wrong_tokens = 0
    accuracy = 0
    known_accuracy = 0
    novel_accuracy = 0
    # test_tokens = 0
    for idx in range(0, len(backpointers)):
        predicted_tag = backpointers[idx]
        observed_word, observed_tag = test[idx].split('/')[0], test[idx].split('/')[1]
        if observed_word != '###':
            # test_tokens += 1
            if predicted_tag == observed_tag:
                correct_test_tokens += 1
                if observed_word in tag_dictionary:
                    known_correct_tokens += 1
                else:
                    novel_correct_tokens += 1
            else:
                wrong_test_tokens += 1
                if observed_word in tag_dictionary:
                    known_wrong_tokens += 1
                else:
                    novel_wrong_tokens += 1    
    if (correct_test_tokens + wrong_test_tokens != 0):
        accuracy = 100 * 1.0 * float(correct_test_tokens) / (correct_test_tokens + wrong_test_tokens)
    else:
        accuracy = float(0)
    if (known_correct_tokens + known_wrong_tokens != 0):
        known_accuracy = 100 * 1.0 * float(known_correct_tokens) / (known_correct_tokens + known_wrong_tokens)
    else:
        known_accuracy = float(0)
    if (novel_correct_tokens + novel_wrong_tokens != 0):
        novel_accuracy = 100 * 1.0 * float(novel_correct_tokens) / (novel_correct_tokens + novel_wrong_tokens)
    else:
        novel_accuracy = float(0)
    if isPosterior == True:
        print("Tagging accuracy (posterior decoding): {}%    (known: {}%    novel: {}%)".format(round(accuracy, 2), round(known_accuracy, 2), round(novel_accuracy, 2)))
    else:
        print("Tagging accuracy (Viterbi decoding): {}%    (known: {}%    novel: {}%)".format(round(accuracy, 2), round(known_accuracy, 2), round(novel_accuracy, 2)))
    return None


if __name__ == "__main__":
    lambda_value = 1 # smooth lambda
    read_file(train)
    compute_perplexity()
    backpointer_list = viterbi_tagging()
    compute_accuracy(backpointer_list, False)
    posterior_probs = forward_backward_tagging()
    compute_accuracy(posterior_probs, True)
