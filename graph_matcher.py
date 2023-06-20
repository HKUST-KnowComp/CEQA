import networkx as nx
from random import sample, choice, random, randint
import numpy as np
import sys
from collections import defaultdict
import pandas as pd

test_query_type = "(i,(i,(p,(p,(e))),(p,(p,(e)))),(p,(e)))"



def count_edges(graph):
    edges = 0
    for node in graph.nodes():
        edges += len(graph[node])
    return edges

def pattern_to_query(pattern, graph):
    while True:
        random_node = sample(list(graph.nodes()), 1)[0]
        # print(random_node)
        query, nl_query = _pattern_to_query(pattern, graph, random_node)
        if query is not None:
            return query, nl_query 


def reverse_relation_name(relation_type):
    if "inversed" in relation_type:
        return relation_type[:-9]
    else:
        return relation_type + " inversed"


def isReversedEdge(relation_type):
    if "inversed" in relation_type:
        return True
    else:
        return False


def _pattern_to_query(pattern, graph, node):
    """
    In this function, _pattern_to_query is recursively used for finding the anchor nodes and relations from
    a randomly sampled entity, which is assumed to be the answer.

    :param pattern:
    :param graph:
    :param node:
    :return:
    """

    pattern = pattern[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    for ii, character in enumerate(pattern):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(pattern[jj: ii])
            jj = ii + 1

    sub_queries.append(pattern[jj: len(pattern)])

    # print(sub_queries)
    if sub_queries[0] == "p":
        # Sample a neighbor and remember the edge
        # reversely_connected_nodes = [next_node for next_node in list(graph.neighbors(node)) if isReversedEdge(graph.edges[node, next_node]['type'])]

        reversely_connected_nodes = np.array([next_node for next_node in list(graph.predecessors(node))])
      
        if len(reversely_connected_nodes) == 0:
            return None, None

        next_node = choice(reversely_connected_nodes)
        edge_name = reverse_relation_name( choice(list(graph.edges[node, next_node].keys())))

        sub_query, natural_query = _pattern_to_query(sub_queries[1], graph, next_node)
        if sub_query is None:
            return None, None

        return "(p,(" + edge_name + '),' + sub_query + ")", natural_query

    elif sub_queries[0] == "e":
        return "(e,(" + node + "))", node
    elif sub_queries[0] == "u":
        # randomly sample a node
        sub_queries_list = []
        natural_queries_list = []

        random_subquery_index = randint(1, len(sub_queries)-1)

        for i in range(1, len(sub_queries)):
            if i == random_subquery_index:
                sub_q, n_q = _pattern_to_query(sub_queries[i], graph, node)
            else:
                sub_q, n_q = _pattern_to_query(sub_queries[1], graph, sample(list(graph.nodes()), 1)[0])
            sub_queries_list.append(sub_q)
            natural_queries_list.append(n_q)

        for sub_query in sub_queries_list:
            if sub_query is None:
                return None, None

        for index_i, sub_query_i in enumerate(sub_queries_list):
            for index_j in range(index_i + 1, len(sub_queries_list)):
                if sub_query_i == sub_queries_list[index_j]:
                    return None, None

        #
        # sub_query_1, natural_query_1 = _pattern_to_query(sub_queries[1], graph, sample(list(graph.nodes()), 1)[0])
        # sub_query_2, natural_query_2 = _pattern_to_query(sub_queries[2], graph, node)
        #
        # if sub_query_1 is None or sub_query_2 is None:
        #     return None, None
        return_str = "(u"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return_natural_query = " and ".join(natural_queries_list)

        return return_str, return_natural_query

    elif sub_queries[0] == "i":

        sub_queries_list = []
        natural_queries_list = []
        for i in range(1, len(sub_queries)):
            sub_q, n_q = _pattern_to_query(sub_queries[i], graph, node)
            sub_queries_list.append(sub_q)
            natural_queries_list.append(n_q)

        for sub_query in sub_queries_list:
            if sub_query is None:
                return None, None

        for index_i, sub_query_i in enumerate(sub_queries_list):
            for index_j in range(index_i+1, len(sub_queries_list)):
                if sub_query_i == sub_queries_list[index_j]:
                    return None, None


        return_str = "(i"
        for sub_query in sub_queries_list:
            return_str += ","
            return_str += sub_query
        return_str += ")"

        return_natural_query = ", or ".join(natural_queries_list)
      

        return return_str, return_natural_query
    else:
        print("Invalid Pattern")
        print(sub_queries)
        exit()


def query_search_answer(graph, query):
    """

    :param graph:
    :param query:
    :param node:
    :return:
    """

    query = query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    # print(sub_queries)
    if sub_queries[0] == "p":

        sub_query_answers, sub_query_explanations, sub_query_explanation_tuples = query_search_answer(graph, sub_queries[2])

        all_answers = []
        all_explanations = []
        all_explanations_tuple = []

        for answer_id, sub_answer in enumerate(sub_query_answers):
            try:
                next_nodes = list(graph.neighbors(sub_answer))
            
            except:
                continue
            explanations = sub_query_explanations[answer_id]
            explanation_tuples = sub_query_explanation_tuples[answer_id]
            
            for node in next_nodes:
                if  sub_queries[1][1:-1] in graph.edges[sub_answer, node].keys():
                # if graph.edges[sub_answer, node]['type'] == sub_queries[1][1:-1]:
                    all_answers.append(node)

                    this_explanation = triple2text_aser(sub_answer,node,sub_queries[1][1:-1])
                    this_explanation_tuple = (sub_answer,node,sub_queries[1][1:-1])

                    previouse_explanations = explanations[:]
                    previouse_explanations.append(this_explanation)

                    previouse_explanation_tuples = explanation_tuples[:]
                    previouse_explanation_tuples.append(this_explanation_tuple)

                    all_explanations.append(previouse_explanations)
                    all_explanations_tuple.append(previouse_explanation_tuples)
                    
        return all_answers, all_explanations, all_explanations_tuple

    elif sub_queries[0] == "e":
        return [sub_queries[1][1:-1]], [[]] , [[]]

    elif sub_queries[0] == "u":

        sub_query_answers_list = []
        explanations_list = []
        explanation_tuples_list = []

        for i in range(1, len(sub_queries)):
            sub_query_answers_i, explanations_i, sub_query_explanation_tuples_i = query_search_answer(graph, sub_queries[i])
            sub_query_answers_list.append(sub_query_answers_i)
            explanations_list.append(explanations_i)
            explanation_tuples_list.append(sub_query_explanation_tuples_i)

        sub_query_dict_dict = {}
        sub_query_dict_dict_tuples = {}

        for i in range(len(explanations_list)):
            for index_a, a in enumerate(sub_query_answers_list[i]):
                sub_query_dict_dict[a] = explanations_list[i][index_a]
                sub_query_dict_dict_tuples[a] = explanation_tuples_list[i][index_a]

        merged_answers = set(sub_query_answers_list[0])
        for sub_query_answers in sub_query_answers_list:
            merged_answers = merged_answers | set(sub_query_answers)


        merged_answers = list(merged_answers)

        merged_explanations = []
        merged_explanation_tuples = []

        for answer in merged_answers:
            answer_explanation = sub_query_dict_dict[answer] 
            merged_explanations.append(answer_explanation)

            answer_explanation_tuples = sub_query_dict_dict_tuples[answer]
            merged_explanation_tuples.append(answer_explanation_tuples)



        return merged_answers, merged_explanations, merged_explanation_tuples


    elif sub_queries[0] == "i":

        sub_query_answers_list = []
        explanations_list = []
        explanation_tuples_list = []

        for i in range(1, len(sub_queries)):

            sub_query_answers_i, explanations_i, sub_query_explanation_tuples_i = query_search_answer(graph, sub_queries[i])
            sub_query_answers_list.append(sub_query_answers_i)
            explanations_list.append(explanations_i)
            explanation_tuples_list.append(sub_query_explanation_tuples_i)


        sub_query_dict_list = []
        sub_query_dict_list_tuples = []
        for i in range(len(explanations_list)):
            sub_query_dict_list.append({a: explanations_list[i][index_a] for index_a, a in enumerate(sub_query_answers_list[i])})
            sub_query_dict_list_tuples.append({a: explanation_tuples_list[i][index_a] for index_a, a in enumerate(sub_query_answers_list[i])})

        merged_answers = set(sub_query_answers_list[0])
        for sub_query_answers in sub_query_answers_list:
            merged_answers = merged_answers & set(sub_query_answers)

        merged_answers = list(merged_answers)

        merged_explanations = []
        merged_explanation_tuples = []
        for answer in merged_answers:
            answer_explanation = [exp  for dic in  sub_query_dict_list for exp in dic[answer]]
            merged_explanations.append(answer_explanation)
            merged_explanation_tuple = [exp  for dic in  sub_query_dict_list_tuples for exp in dic[answer]]
            merged_explanation_tuples.append(merged_explanation_tuple)


        return merged_answers, merged_explanations, merged_explanation_tuples
    else:
        print("Invalid Pattern")
        print(sub_queries)
        exit()



def get_anchor_nodes(query):
    anchor_nodes = []
    
    # print(query)
    query = query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    # print(sub_queries)
    if sub_queries[0] == "p":

        return get_anchor_nodes(sub_queries[2])

    elif sub_queries[0] == "e":
        return [sub_queries[1][1:-1]]

    elif sub_queries[0] == "u" or sub_queries[0] == "i":

        anchor_nodes = []

        for i in range(1, len(sub_queries)):
            sub_anchor_nodes = get_anchor_nodes(sub_queries[i])
            anchor_nodes.extend(sub_anchor_nodes)

        return anchor_nodes

  
    else:
        print("Invalid Pattern")
        print(sub_queries)
        exit()






def triple2text_aser(head_str, tail_str, feat_key):


    if isReversedEdge(feat_key):

        tmp = head_str
        head_str = tail_str
        tail_str = tmp

        feat_key = reverse_relation_name(feat_key)


    if feat_key == "Conjunction":
        triple_interpretation = head_str + ", and " + tail_str + "."

    elif feat_key == "Synchronous":
        triple_interpretation = head_str + ", and at the same time " + tail_str + "."

    elif feat_key == "Instantiation":
        triple_interpretation = head_str + ", for example " + tail_str + "."

    elif feat_key == "Restatement":
        triple_interpretation = tail_str + ", and in other words, " + head_str + "."

    elif feat_key == "Alternative":
        triple_interpretation = head_str + ", or " + tail_str + "."

    elif feat_key == "ChosenAlternative":
        triple_interpretation = head_str + ", instead of " + tail_str + "."
        

    elif feat_key == "Exception":
        # triple_interpretation = tail_str + " is an exception of " + head_str + "."
        triple_interpretation = head_str + ", except " + tail_str + "."


    elif feat_key == "Contrast":
        # triple_interpretation = head_str + " and " + \
                                # tail_str + " share significant difference regarding some property."

        triple_interpretation = head_str + ", but " + tail_str + "."

    elif feat_key == "Concession":
        triple_interpretation = head_str + ", although " + tail_str + "."


    elif feat_key == "Condition":
        triple_interpretation = head_str +  ", if " + tail_str + "."

    elif feat_key == "Precedence":
        triple_interpretation = head_str + " before "  + tail_str + "."

    elif feat_key == "Succession":
        triple_interpretation = head_str + " after " + tail_str + "."

    elif feat_key == "Result":
        triple_interpretation = head_str + ", therefore " + tail_str + "."

    elif feat_key == "Reason":
        triple_interpretation = head_str + ", because " + tail_str + "."

    else:
        triple_interpretation = head_str + " " + tail_str + "."

    return triple_interpretation



def interpretateQueryDirectly(query, shared_var, triple2text=triple2text_aser):

    query = query[1:-1]
    parenthesis_count = 0

    sub_queries = []
    jj = 0

    

    for ii, character in enumerate(query):
        # Skip the comma inside a parenthesis
        if character == "(":
            parenthesis_count += 1

        elif character == ")":
            parenthesis_count -= 1

        if parenthesis_count > 0:
            continue

        if character == ",":
            sub_queries.append(query[jj: ii])
            jj = ii + 1

    sub_queries.append(query[jj: len(query)])

    if sub_queries[0] == "p":

        shared_var["counter"] += 1
        this_variable_index = shared_var["counter"]

        sub_nl_query, isAnchor, sub_variable_index = interpretateQueryDirectly(sub_queries[2], shared_var, triple2text)

        if isAnchor:
            head_str = sub_nl_query
        
        else:
            head_str = "V" + str(sub_variable_index)
        

        this_query_relation = sub_queries[1][1:-1]

        tail_str = "V" + str(this_variable_index)

        

        this_nl_edge = triple2text(head_str, tail_str, this_query_relation)

        if isAnchor:
            # print(this_nl_edge)
            return this_nl_edge, False, this_variable_index

        else:
            # print(sub_nl_query + " " + this_nl_edge)
            return sub_nl_query + " " + this_nl_edge, False, this_variable_index


    elif sub_queries[0] == "e":
        # print(sub_queries[1][1:-1])
        return sub_queries[1][1:-1], True, shared_var["counter"]


    elif sub_queries[0] == "i":

        shared_var["counter"] += 1
        this_variable_index = shared_var["counter"]

        this_nl_edges = "V" + str(this_variable_index) + " represents all of "
        sub_queries_text = ""
        variables_list = []
        for i in range(1, len(sub_queries)):
            sub_nl_query, isAnchor, sub_variable_index = interpretateQueryDirectly(sub_queries[i], shared_var, triple2text)
            
            sub_queries_text += sub_nl_query + " "

            head_str = "V" + str(sub_variable_index)
        
            variables_list.append(head_str)

        variables_text = this_nl_edges + ", ".join(variables_list) + "."

        return sub_queries_text +  variables_text, False, this_variable_index
    
    elif sub_queries[0] == "u":

        shared_var["counter"] += 1
        this_variable_index = shared_var["counter"]

        this_nl_edges = "V" + str(this_variable_index) + " represents any of "
        sub_queries_text = ""
        variables_list = []
        for i in range(1, len(sub_queries)):
            sub_nl_query, isAnchor, sub_variable_index = interpretateQueryDirectly(sub_queries[i], shared_var, triple2text)
            
            sub_queries_text += sub_nl_query + " "

            head_str = "V" + str(sub_variable_index)
        
            variables_list.append(head_str)

        variables_text = this_nl_edges + ", ".join(variables_list) + "."

        return sub_queries_text +  variables_text, False, this_variable_index

    else:
        print("Invalid Pattern")
        exit()



def sample_cskg(input_graph, _triple2text):

    all_query_types = pd.read_csv("./test_generated_formula_anchor_node=3.csv").reset_index(
        drop=True)  # debug

    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]
        if  "n" in query or "u" in query:
            continue
        original_query_types[fid] = query
        


    for query_name, query_type in original_query_types.items():
        print("=====================================")
        print(query_name)
        print(query_type)
        query, nl_query = pattern_to_query(query_type, input_graph)
        print(query)

        shared={"counter": -1}
        
        query_interpretation = interpretateQueryDirectly(query, shared_var=shared, triple2text=_triple2text)
        print(query_interpretation)

        answers, explanations, explanation_tuples = query_search_answer(input_graph, query)
        assert len(answers) == len(explanations) == len(explanation_tuples)
        print("answers[:5]: ", answers[:5])
        print(explanations[:5])
        
        print(explanation_tuples[:5])

        more_background_tuples = extend_query(input_graph, query)
        print("anchor nodes: ", get_anchor_nodes(query))
        print("more background tuples: ", more_background_tuples)

        occurential_tuples =  extend_occurential_constraints(input_graph, explanation_tuples)
        print("occurential_tuples: ", occurential_tuples)

        temporal_tuples = extend_temporal_constraints(input_graph, explanation_tuples)
        print("temporal_tuples: ", temporal_tuples)



def extend_query(graph, query):
    """
    Extend the query by adding more edges
    :param graph: the graph
    :param query: the query
    :return: the extended query
    """
    # print(query)
    anchor_nodes = get_anchor_nodes(query)
    
    extended_tuple = []
    for anchor_node in anchor_nodes:
        # print(anchor_node)
        # print(graph.neighbors(anchor_node))
        for i in range(3):
            neighbor = choice([n for n in  graph.neighbors(anchor_node)])
            relation_type = choice(list(graph[anchor_node][neighbor].keys()))
            extended_tuple.append((anchor_node, neighbor, relation_type))
    extended_tuple = list(set(extended_tuple))

  
    return extended_tuple




def extend_occurential_constraints(graph, all_explanation_tuples):
    """
    Extend the explanation tuples by adding more edges
    This explanations are used to extend the situations so that some answers are no longer correct.



    :param graph: the graph
    :param explanation_tuples: the explanation tuples
    :return: the extended explanation tuples
    """
    
    if len(all_explanation_tuples) == 0:
        return []

    all_must_hold_eventualities = []
    all_must_not_hold_eventualities = []

    head_must_occur_relations = ["Conjunction", "Instantiation","Synchronous", "ChosenAlternative", 
                                 "Contrast", "Concession", "Precedence", "Succession", "Result", "Reason"]
    head_must_not_occur_relations = ["Exception"]
    tail_must_occur_relations = ["Conjunction", "Synchronous", "Synchronous", "Exception", 
                                 "Contrast", "Concession", "Precedence", "Succession", "Result", "Reason"]
    tail_must_not_occur_relations = ["ChosenAlternative"]
    
    
    inverse_head_must_occur_relations = [reverse_relation_name(r) for r in  tail_must_occur_relations ]
    inverse_tail_must_occur_relations = [reverse_relation_name(r) for r in  head_must_occur_relations ]

    inverse_head_must_not_occur_relations = [reverse_relation_name(r) for r in  tail_must_not_occur_relations ]
    inverse_tail_must_not_occur_relations = [reverse_relation_name(r) for r in  head_must_not_occur_relations ]

    head_must_occur_relations += inverse_head_must_occur_relations
    tail_must_occur_relations += inverse_tail_must_occur_relations
    head_must_not_occur_relations += inverse_head_must_not_occur_relations
    tail_must_not_occur_relations += inverse_tail_must_not_occur_relations


    head_must_occur_relations = set(head_must_occur_relations)
    tail_must_occur_relations = set(tail_must_occur_relations)
    head_must_not_occur_relations = set(head_must_not_occur_relations)
    tail_must_not_occur_relations = set(tail_must_not_occur_relations)

    # print("head_must_occur_relations: ", head_must_occur_relations)
    # print("tail_must_occur_relations: ", tail_must_occur_relations)
    # print("head_must_not_occur_relations: ", head_must_not_occur_relations)
    # print("tail_must_not_occur_relations: ", tail_must_not_occur_relations)

    for explanation_tuples in all_explanation_tuples:
        must_hold_eventualities = []
        must_not_hold_eventualities = []
        for explanation_tuple in explanation_tuples:
            head, tail, relation = explanation_tuple
            if relation in head_must_occur_relations:
                must_hold_eventualities.append(head)
            if relation in tail_must_occur_relations:
                must_hold_eventualities.append(tail)
            if relation in head_must_not_occur_relations:
                must_not_hold_eventualities.append(head)
            if relation in tail_must_not_occur_relations:
                must_not_hold_eventualities.append(tail)
        
        must_hold_eventualities = list(set(must_hold_eventualities))
        must_not_hold_eventualities = list(set(must_not_hold_eventualities))



        all_must_hold_eventualities.append(must_hold_eventualities)
        all_must_not_hold_eventualities.append(must_not_hold_eventualities)

    must_hold_eventualities_in_common = set(all_must_hold_eventualities[0])
    for must_hold_eventualities in all_must_hold_eventualities[1:]:
        must_hold_eventualities_in_common = must_hold_eventualities_in_common.intersection(set(must_hold_eventualities))
    
    must_not_hold_eventualities_in_common = set(all_must_not_hold_eventualities[0])
    for must_not_hold_eventualities in all_must_not_hold_eventualities[1:]:
        must_not_hold_eventualities_in_common = must_not_hold_eventualities_in_common.intersection(set(must_not_hold_eventualities))

    
    unique_must_hold_eventualities = []
    for must_hold_eventualities in all_must_hold_eventualities:
        unique_must_hold_eventualities.extend(list(set(must_hold_eventualities) - must_hold_eventualities_in_common))
    
    unique_must_not_hold_eventualities = []
    for must_not_hold_eventualities in all_must_not_hold_eventualities:
        unique_must_not_hold_eventualities.extend(list(set(must_not_hold_eventualities) - must_not_hold_eventualities_in_common))

    unique_must_hold_eventualities = list(set(unique_must_hold_eventualities))
    unique_must_not_hold_eventualities = list(set(unique_must_not_hold_eventualities))


    # print("all_must_hold_eventualities: ", all_must_hold_eventualities)
    # print("all_must_not_hold_eventualities: ", all_must_not_hold_eventualities)

    # print("must_hold_eventualities_in_common: ", must_hold_eventualities_in_common)
    # print("must_not_hold_eventualities_in_common: ", must_not_hold_eventualities_in_common)

    # print("unique_must_hold_eventualities: ", unique_must_hold_eventualities)
    # print("unique_must_not_hold_eventualities: ", unique_must_not_hold_eventualities)

    if len(unique_must_hold_eventualities) == 0 and len(unique_must_not_hold_eventualities) == 0:
        return []

    occurential_critical_edges = []

    for must_hold_eventuality in unique_must_hold_eventualities:
        for neighbour in graph.successors(must_hold_eventuality):
            # print("neighbour: ", neighbour)
            # print("graph[must_hold_eventuality][neighbour]: ", graph[must_hold_eventuality][neighbour])
            for rel in head_must_not_occur_relations:
                if rel in graph[must_hold_eventuality][neighbour]:
                    occurential_critical_edges.append((must_hold_eventuality, neighbour, rel))
                    occurential_critical_edges.append((neighbour, must_hold_eventuality, reverse_relation_name(rel)))
    
    for must_not_hold_eventuality in unique_must_not_hold_eventualities:
        for neighbour in graph.successors(must_not_hold_eventuality):
            for rel in head_must_occur_relations:
                if rel in graph[must_not_hold_eventuality][neighbour]:
                    occurential_critical_edges.append((must_not_hold_eventuality, neighbour, rel))
                    occurential_critical_edges.append((neighbour, must_not_hold_eventuality, reverse_relation_name(rel)))
    

    return occurential_critical_edges 



def extend_temporal_constraints(graph, all_explanation_tuples):

    if len(all_explanation_tuples) == 0:
        return []

    all_temporal_constraints = []

    for explanation_tuples in all_explanation_tuples:
        have_precedent_dict = {}

        for explanation_tuple in explanation_tuples:
            head, tail, relation = explanation_tuple
            if relation in ["Precedence", "Succession inversed", "Result", "Reason inversed", "Condition inversed"]:
                if tail in have_precedent_dict:
                    have_precedent_dict[tail].append(head)
                else:
                    have_precedent_dict[tail] = [head]
            
            if relation in ["Succession", "Precedence inversed", "Reason", "Result inversed", "Condition"]:
                if head in have_precedent_dict:
                    have_precedent_dict[head].append(tail)
                else:
                    have_precedent_dict[head] = [tail]

        if len(have_precedent_dict) == 0:
            return []

        for i in range(2):
            for key in have_precedent_dict:
                for precedent in have_precedent_dict[key]:
                    if precedent in have_precedent_dict:
                        have_precedent_dict[key].extend(have_precedent_dict[precedent])
                        have_precedent_dict[key] = list(set(have_precedent_dict[key]))
        
        termporal_contradictory_edges = []
        for e, prev_e_list in have_precedent_dict.items():
            for prev_e in prev_e_list:
                if prev_e in graph.successors(e):
                    for rel in graph[e][prev_e]:
                        if rel in ["Precedence", "Succession inversed", "Result", "Reason inversed", "Condition inversed"]:
                            termporal_contradictory_edges.append((e, prev_e, rel))
                            termporal_contradictory_edges.append((prev_e, e, reverse_relation_name(rel)))
        
        all_temporal_constraints.append(termporal_contradictory_edges)
    
    termporal_constraints_in_common = set(all_temporal_constraints[0])
    for termporal_constraints in all_temporal_constraints[1:]:
        termporal_constraints_in_common = termporal_constraints_in_common.intersection(set(termporal_constraints))
    

    unique_temporal_constraints = []
    for termporal_constraints in all_temporal_constraints:
        unique_temporal_constraints.extend(list(set(termporal_constraints) - termporal_constraints_in_common))
    
    unique_temporal_constraints = list(set(unique_temporal_constraints))

    return unique_temporal_constraints
        





if __name__ == "__main__":
    

    aser_graph = nx.read_gpickle("/home/data/jbai/aser_graph/aser50k_train.pickle")
    sample_cskg(aser_graph, triple2text_aser)



