#!/usr/bin/env python3

import copy
import sys
import math

def prob_to_weight(prob):
    prob = float(prob)
    return -1 * math.log2(prob)

def print_tree(columns, slot, output, isComplete):
    if isComplete and len(slot) > 4:
        output.append("(" + grammar_table[slot[0]][0])

    if len(slot) > 4:
        left_child = []
        right_child = []
        for s in columns[slot[4][0]]:
            if s is None:
                continue
            if s[0] == slot[4][1] and s[2] == slot[4][3] and s[3] == slot[4][4]:
                left_child = s
                break
        if slot[5][2] < 0:
            right_child = slot[5][1:]
        else:
            for s in columns[slot[5][0]]:
                if s is None:
                    continue
                if s[0] == slot[5][1] and s[2] == slot[5][3] and s[3] == slot[5][4]:
                    right_child = s
                    break
        if len(left_child) != 0:
            print_tree(columns, left_child, output, False)
        if len(right_child) != 0:
            print_tree(columns, right_child, output, True)    

    if isComplete:
        if len(slot) == 4:
            output.append(slot[0])
        else:
            output.append(")")


def process_grammar(grammar, initial_values, grammar_table, non_terminals, initial_rules, all_rules):
    for line in grammar.readlines():
        line = line.strip().split()
        weight = prob_to_weight(line[0])
        rule = line[1:]
        if rule[0] not in non_terminals:
            interned_nonterminal = sys.intern(rule[0])
            non_terminals.append(rule[0])
        if rule[0] == 'ROOT':
            initial_rules.append(rule)
        interned_rule = sys.intern(' '.join(rule))
        if interned_rule not in grammar_table:
            # rules are converted from list to string (sep by space) to be interned in dict
            # remember to compare or retrive rules from hash table, do convert the keys back to list
            grammar_table[interned_rule] = rule
            initial_values[interned_rule] = weight
        # tp = (rule[0], rule)
        # all_rules.append(tp)
        if rule[0] not in all_rules:
            all_rules[rule[0]] = []
            all_rules[rule[0]].append(rule)
        else:
            all_rules[rule[0]].append(rule)


def parse_sentence(sentence, initial_values, grammar_table, non_terminals, initial_rules, all_rules):

    # Initializtion
    initial_rows = []
    for initial_rule in initial_rules:
        initial_weight = initial_values[' '.join(initial_rule)]
        backpointer = 0
        initial_dot_pos = 0
        interned_initial_rule = sys.intern(' '.join(initial_rule))
        initial_rows.append([interned_initial_rule, initial_weight, backpointer, initial_dot_pos])

    sentence = sentence.split()
    customer_columns = []
    columns = [] # columns of parse table
    for i in range(0, len(sentence) + 1):
        columns.append({}) # MODIFICATION: change each column into a hash table: 
                           #               key: first elem of rule (head), value: rules sharing the same head
        customer_columns.append({})

    for initial_row in initial_rows:
        if grammar_table[initial_row[0]][0] not in columns[0]:
            columns[0][grammar_table[initial_row[0]][0]] = [initial_row]
            
        else:
            columns[0][grammar_table[initial_row[0]][0]].append(initial_row)

    for col in range(0, len(sentence) + 1):
        # remember to modulize attach, predict, and scan functions later
        idx = 0
        col_size = 0
        queue = []
        key_list = []
        checked = [] # prevent lookaheads that have been checked from being checked again
        duplicate = {} # hash table checking duplicates for each column. key: str(rule), value: idx
        for head in columns[col].keys():
            for slot in columns[col][head]:
                # col_size += 1
                queue.append(head)
                if head not in key_list:
                    key_list.append(head)
        col_size = len(queue)
        sub_index = 0
        while idx < col_size:
            # attach)
            if idx == 0:
                sub_index = 0
            else:
                if queue[idx] != queue[idx - 1]:
                    sub_index = 0
                else:
                    sub_index += 1

            if columns[col][queue[idx]][sub_index] is None:
                idx += 1
                continue
            cur_rule = columns[col][queue[idx]][sub_index][0]
            # print("cur_rule: ", cur_rule)
            # print("Comparison:", len(grammar_table[cur_rule]), columns[col][queue[idx]][sub_index][3] + 1)
            # print()            
            if len(grammar_table[cur_rule]) == columns[col][queue[idx]][sub_index][3] + 1:
                # handling duplicate, esepcially at the last column
                # for each column, set up a temporary hash table to store interned rule-idx pair, if the newly added item has a match, return its idx to access the duplicate item.
                # Compare start position first, if yes then weights, if current weight is larger than the duplicate one, don't add it to the column 
                # find the start position of the completed rule, go back to the column where the rule starts, append each of the related to current column
                '''
                for head in key_list:
                    for constituent in columns[col][head]:
                        if constituent is None:
                            idx += 1
                            continue
                '''
                        # cur_rule = grammar_table[constituent[0]] 
                        # for slot in customer_columns[constituent[2]][head]: # gist of the change!!!!!!!!!!
                for head in columns[columns[col][queue[idx]][sub_index][2]]:
                    for slot in columns[columns[col][queue[idx]][sub_index][2]][head]:
                        if slot is None:
                            continue                        
                        cur_slot_rule = grammar_table[slot[0]]
                        if len(cur_slot_rule) - 1 > slot[3]: 
                            if cur_slot_rule[slot[3] + 1] == grammar_table[columns[col][queue[idx]][sub_index][0]][0]: # match lookahead with first node
                                attached = copy.deepcopy(slot)
                                attached[3] += 1
                                attached[1] += columns[col][queue[idx]][sub_index][1] # update the weight of the newly attached rule
                                if len(attached) > 4:
                                    attached = attached[:-2]
                                attached.append([columns[col][queue[idx]][sub_index][2], slot[0], slot[1], slot[2], slot[3]])
                                attached.append([col, columns[col][queue[idx]][sub_index][0], columns[col][queue[idx]][sub_index][1], columns[col][queue[idx]][sub_index][2], columns[col][queue[idx]][sub_index][3]])                            
                                interned = sys.intern(' '.join(grammar_table[attached[0]]) + " " + str(attached[2]) + " " + str(attached[3]))
                                if interned not in duplicate:
                                    if grammar_table[attached[0]][0] not in key_list:
                                        columns[col][grammar_table[attached[0]][0]] = [attached] # attach under new head key
                                        key_list.append(grammar_table[attached[0]][0])
                                    else:
                                        columns[col][grammar_table[attached[0]][0]].append(attached) # attach
                                    duplicate[interned] = len(columns[col][queue[idx]]) - 1
                                    queue.append(grammar_table[attached[0]][0])
                                    col_size += 1

                                else: # detecting duplicate rule
                                    print()
                                    print("Compared check at col ", col, "idx ", idx)
                                    print(columns[col][grammar_table[attached[0]][0]])
                                    print(duplicate)
                                    print(interned)
                                    print(duplicate[interned])
                                    compared = columns[col][grammar_table[attached[0]][0]][duplicate[interned]] # access the slot of the duplicate rule
                                    if attached[1] < compared[1]:
                                        columns[col][grammar_table[attached[0]][0]][duplicate[interned]] = None #  kill the heavier one, and add the new one. Otherwise, don't append the new one.
                                        columns[col][grammar_table[attached[0]][0]].append(attached)
                                        duplicate[interned] = len(columns[col][queue[idx]]) - 1
                                        queue.append(grammar_table[attached[0]][0])
                                        col_size += 1
                                        
            else:
                cur_dot_pos = columns[col][queue[idx]][sub_index][3]

                if grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1] in non_terminals: # PREDICT if the look-ahead is a nonterminal
                    if grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1] not in checked: # no checked lookahead is processed again
                        checked.append(grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1])
                        next_node = grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1]
                        if next_node in all_rules:
                            for rule in all_rules[next_node]:
                                row = [sys.intern(' '.join(rule)), initial_values[' '.join(rule)], col, 0]
                                if next_node not in key_list:
                                    columns[col][next_node] = []
                                    columns[col][next_node].append(row)
                                    key_list.append(next_node)                                   
                                else:
                                    columns[col][next_node].append(row) # append the slot          
                                duplicate[sys.intern(' '.join(grammar_table[row[0]])) + " " + str(row[2]) + " " + str(row[3])] = sub_index
                                queue.append(next_node)
                                col_size += 1
            
                else: # SCAN if the look-ahead is a word
                    if col < len(sentence):
                        # print("lookahead: ", grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1])
                        # print("current word: ", sentence[col])
                        # print("comparison: ", grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1], sentence[col])
                        if grammar_table[columns[col][queue[idx]][sub_index][0]][cur_dot_pos + 1] == sentence[col]:
                            temp = copy.deepcopy(columns[col][queue[idx]][sub_index])
                            temp[3] += 1

                            if len(temp) > 4:
                                temp = temp[:-2]
                            temp.append([col, columns[col][queue[idx]][sub_index][0], columns[col][queue[idx]][sub_index][1], columns[col][queue[idx]][sub_index][2], columns[col][queue[idx]][sub_index][3]])
                            temp.append([col, sys.intern(sentence[col]), -1.0, columns[col][queue[idx]][sub_index][2], 0])
                            if grammar_table[temp[0]][0] not in columns[col + 1]:
                                # print("scan and update next column", grammar_table[temp[0]][0])
                                columns[col + 1][grammar_table[temp[0]][0]] = [temp]
                            else:
                                columns[col + 1][grammar_table[temp[0]][0]].append(temp)
            # print("idx: ", idx)
            idx += 1
        print("print queue ", col)
        print(queue)
        print()
        # we'll deal with this later
        '''
        for nont in columns[col]:
            for rl in columns[col][nont]:
                if len(grammar_table[rl[0]]) > rl[3] + 1:
                    # print("lookahead", grammar_table[rl[0]][rl[3] + 1])
                    if grammar_table[rl[0]][rl[3] + 1] not in customer_columns[col]:
                        customer_columns[col][grammar_table[rl[0]][rl[3] + 1]] = [rl]
                    else:
                        customer_columns[col][grammar_table[rl[0]][rl[3] + 1]].append(rl)
        '''
        
    return columns


if __name__ == "__main__":

    grammar = open(sys.argv[1])
    sentences = open(sys.argv[2])

    initial_values = {} # key: str(rule), value: weight
    grammar_table = {}  # key: str(rule), value: rule(list of tokens)
    # nonterminal_rules = [] # list of tuples, key: nonterminal, value; rule(list of tokens)   
    all_rules = {}         # hash table of tuples, key: nonterminal, value: a list of rule(list of tokens)
    initial_rules = []   # list of initial rule, for initialization of columns
    non_terminals = []  # list of nonterminals

    process_grammar(grammar, initial_values, grammar_table, non_terminals, initial_rules, all_rules)

    for sentence in sentences.readlines():
        if sentence == "\n":
            continue
        columns = parse_sentence(sentence, initial_values, grammar_table, non_terminals, initial_rules, all_rules)

        idx = 0
        for i in columns[1]:
            for y in columns[1][i]:
                idx += 1
        # print("Last column:")
        noParse = True
        # for row in columns[-1]:
        #     if row is None:
        #         continue
        #     output = []
        #     if grammar_table[row[0]][0] == 'ROOT' and row[3] == len(grammar_table[row[0]]) - 1:
        #         noParse = False
        #         print_tree(columns, row, output, True)
        #         print(' '.join(output))
        #         print(row[1])
    
        for key in columns[-1].keys():
            if key is None:
                continue
            output = []
            if key == 'ROOT':
                for completed_rule in columns[-1][key]:
                    if completed_rule[3] == len(grammar_table[completed_rule[0]]) - 1:
                        noParse = False
                        print_tree(columns, completed_rule, output, True)
                        print(' '.join(output))
                        print(completed_rule[1])

        if noParse:
            print("NONE")
