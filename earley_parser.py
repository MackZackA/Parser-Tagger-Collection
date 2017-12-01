#!/usr/bin/env python3

import copy
import sys
import math

grammar = open(sys.argv[1])
sentences = open(sys.argv[2])


def prob_to_weight(prob):
    prob = float(prob)
    return -1 * math.log2(prob)

initial_values = {} # key: str(rule), value: weight
grammar_table = {}  # key: str(rule), value: rule(list of tokens)
# nonterminal_rules = [] # list of tuples, key: nonterminal, value; rule(list of tokens)   
all_rules = []         # list of tuples, key: nonterminal, value: rule(list of tokens)
initial_rule = []   # list of initial rule, for initialization of columns
non_terminals = []  # list of nonterminals


def print_tree(columns, slot):
    output = ''
    if len(slot) == 3:
        output = '(' + ' '.join(slot[0]) + ')'
    else:
        idx = 3
        output +='('
        while idx < len(slot):
            child = []
            if columns[slot[idx][0]][0] == slot[idx][1] and columns[slot[idx][0]][2] == slot[idx][3]:
                child = columns[slot[idx][0]]
                break
            output += ' '.join(slot[0]) + print_tree(columns, child)
            idx += 1

        output += ')'
        
        '''
        left_child = []
        right_child = []
        for s in columns[slot[-2][0]]:
            if s[0] == slot[-2][1] and s[2] == slot[-2][3]:
                left_child = s
                break
        for s in columns[slot[-1][0]]:
            if s[0] == slot[-1][1] and s[2] == slot[-1][3]:
                right_child = s
                break
        output += "(" + ' '.join(slot[0]) + print_tree(columns, left_child) + print_tree(columns, right_child) + ")"
        '''

    return output



for line in grammar.readlines():
    # print('line before splitting: ', line)
    line = line.strip().split()
    # line = line.strip().split('\t')
    # print('line after splitting: ', line)
    weight = prob_to_weight(line[0])
    rule = line[1:]
    if rule[0] not in non_terminals:
        interned_nonterminal = sys.intern(rule[0])
        non_terminals.append(rule[0])
        # tp = (rule[0], rule)
        # nonterminal_rules.append(tp)
    if rule[0] == 'ROOT':
        initial_rule = rule
    interned_rule = sys.intern(' '.join(rule))
    if interned_rule not in grammar_table:
        # rules are converted from list to string (sep by space) to be interned in dict
        # remember to compare or retrive rules from hash table, do convert the keys back to list
        grammar_table[interned_rule] = rule
        initial_values[interned_rule] = weight
    tp = (rule[0], rule)
    all_rules.append(tp)

'''
print('all_rules:', all_rules)
print()
print('non_terminals:', non_terminals)
print()
print('grammar_table:', grammar_table)
print()
print('initial_values:', initial_values)
print()
print('initial_rule:', initial_rule)
print()
'''


# Initializtion
initial_weight = initial_values[' '.join(initial_rule)]
backpointer = 0
initial_row = [initial_rule, initial_weight, backpointer]

for sentence in sentences.readlines():
    sentence = sentence.split()
# sentence = sentences.read().split()[0]
# print(sentence)
# if 1 > 0:
#    sentence = sentence.split()

    columns = [] # columns of parse table
    for i in range(0, len(sentence) + 1):
        columns.append([])

    columns[0].append(initial_row)
    for col in range(0, len(sentence) + 1):
        # remember to modulize attach, predict, and scan functions later
        col_size = len(columns[col])
        idx = 0

        checked = [] # prevent lookaheads that have been checked from being checked again
        duplicate = {} # hash table checking duplicates for each column. key: str(rule), value: idx
        while idx < col_size:
            
            # print("" + str(idx) + " loop " + str(col) + " col")
            
            # attach
            if len(columns[col][idx][0]) == 1:
                # handling duplicate, esepcially at the last column
                # for each column, set up a temporary hash table to store interned rule-idx pair, if the newly added item has a match, return its idx to access the duplicate item.
                # Compare start position first, if yes then weights, if current weight is larger than the duplicate one, don't add it to the column 


                # find the start position of the completed rule, go back to the column where the rule starts, append each of the related to current column

                # print("At " + str(col)  + " col  idx = " + str(idx) + " attach")
                # count = 1 # indicate how many rules are attached using baclpointers
                for slot in columns[columns[col][idx][2]]:
                    # print("Pointing back to column " + str(columns[col][idx][2]))
                    if len(slot[0]) > 1:
                        if slot[0][1] == columns[col][idx][0][0]:
                            attached = copy.deepcopy(slot)
                            del attached[0][1] # move the dot one position to the right
                            attached[1] += columns[col][idx][1] # update the weight of the newly attached rule
                            
                           #xc
                            # attached.append([columns[col][idx][2], slot[0],slot[2]])
                            # attached.append([col, columns[col][idx][0], columns[col][idx][2]])
                           #/xc
                            # if len(attached) > 3:
                            #     attached = attached[:-2]
                            # attached.append([columns[col][idx][2], slot[0], slot[1], slot[2]])
                            attached.append([col, columns[col][idx][0], slot[1], columns[col][idx][2]])

                            interned = sys.intern(' '.join(attached[0]))
                            if interned not in duplicate:
                                columns[col].append(attached) # attach
                                duplicate[interned] = idx
                                col_size += 1
                                
                                # print("Attach " + str(count) + "th rule for this completed rule")
                                # count += 1
                                # print("The attached slot is ")
                                # print(' '.join(attached[0]) + str(attached[2]))

                            else: # detecting duplicate rule
                                compared = columns[col][duplicate[interned]] # access the slot of the duplicate rule
                                if compared[2] == attached[2]: # they must have the same start position to be duplicates 
                                    if attached[1] < compared[2]:
                                        compared = None #  kill the heavier one, and add the new one. Otherwise, don't append the new one.
                                        columns[col].append(attached)
                                        duplicate[interned] = idx # update the index, since the old one is deleted
                                        col_size += 1

                                        # print("Attach " + str(count) + "th rule for this completed rule")
                                        # print("The attached rule is ") 
                                        # print(' '.join(attached[0]))
                                        # count += 1
            
            else:
                if columns[col][idx][0][1] in non_terminals: # PREDICT if the look-ahead is a nonterminal
                    if columns[col][idx][0][1] not in checked: # no checked lookahead is processed again
                        # print('At ' + str(col) + ' col idx = ' + str(idx) + ' predict')
                        checked.append(columns[col][idx][0][1])
                        # print(checked)
                        # print(columns[col][idx][0])

                        # error here. can't use nonterminal_rules since it doesn't print out all the rules required of the expansion
                        for pair in all_rules:
                            if pair[0] == columns[col][idx][0][1]: # retreive the rule starting with the nonterminal
                                row = [pair[1], initial_values[' '.join(pair[1])], col]
                                columns[col].append(row) # append the slot
                                duplicate[sys.intern(' '.join(row[0]))] = idx
                                col_size += 1
                            # else:
                            #     print("not predicting anything")
                    # else:
                        # print(str(columns[col][idx][0][1]) + " will cause left-recursive rule. Don't add it")

                else: # SCAN if the look-ahead is a word
                    # print('At ' + str(col) + ' col idx = ' + str(idx) + ' scan')
                    if col < len(sentence):
                        if columns[col][idx][0][1] == sentence[col]:
                            temp = copy.deepcopy(columns[col][idx])
                            del temp[0][1]
                            columns[col + 1].append(temp) # append in next line after attaching the word
                            # duplicate[sys.intern(' '.join(temp[0]))] = idx
                        # else:
                            # print("not matched")
            idx += 1
        # print("PRINT column: " + str(col))
        # print(columns[col])

if __name__ == "__main__":
    
    result = ''
    # print("Last column:")
    # for row in columns[-1]:
    #     print(row)
    for row in columns[-1]:
        if row[0][0] == 'ROOT':
            result = print_tree(columns, row)
            print(result)
          
