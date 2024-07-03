import numpy as np
from collections import defaultdict
import json
import networkx as nx

from graph_matcher import  pattern_to_query, query_search_answer, \
triple2text_aser, interpretateQueryDirectly, count_edges, extend_occurential_constraints, extend_temporal_constraints
import pandas as pd
from tqdm import tqdm

from random import choices
import time

MAX_CONSTRAINTS = 10


def sample_general_train(train_graph, query_type, _triple2text, constraint_type):

    while True:

        query, _ = pattern_to_query(query_type, train_graph)


        shared={"counter": -1}
        query_interpretation, _, _ = interpretateQueryDirectly(query, shared_var=shared, triple2text=_triple2text)
    
        answers, explanations, explanation_tuples = query_search_answer(train_graph, query)

        if len(answers) == 0:
            continue

        if constraint_type == "occurential":
            occurential_tuples = extend_occurential_constraints(train_graph, explanation_tuples)
            if len(occurential_tuples) == 0:
                continue
            
            occurential_tuples = choices(occurential_tuples, k=MAX_CONSTRAINTS)
            occurential_tuples = list(set(occurential_tuples))
        
        else:
            occurential_tuples = []
            
        if constraint_type == "temporal":
            temporal_tuples = extend_temporal_constraints(train_graph, explanation_tuples)
            if len(temporal_tuples) == 0:
                continue
            
            temporal_tuples = choices(temporal_tuples, k=MAX_CONSTRAINTS)
            temporal_tuples = list(set(temporal_tuples))
        
        else:
            temporal_tuples = []
        
        
        if len(answers) > 0:
            # background_tuples = extend_query(train_graph, query)

            this_query_result = {
            "query": query,
            "nl_query": query_interpretation + " What is V0?",
            "train_answers": answers,
            "train_explanations": explanations,
            "train_explanation_tuples": explanation_tuples,
            "occurential_tuples": occurential_tuples,
            "temporal_tuples": temporal_tuples
            }
            return this_query_result


def sample_general_valid(train_graph, valid_graph, query_type, _triple2text, constraint_type):

    while True:

        query, _ = pattern_to_query(query_type, valid_graph)

        shared={"counter": -1}
        query_interpretation, _, _ = interpretateQueryDirectly(query, shared_var=shared, triple2text=_triple2text)
    
        train_answers, train_explanations, train_explanation_tuples = query_search_answer(train_graph, query)
        valid_answers, valid_explanations, valid_explanation_tuples = query_search_answer(valid_graph, query)

        if len(valid_answers) == 0 or len(train_answers) == len(valid_answers):
            continue


        if constraint_type == "occurential":
            occurential_tuples = extend_occurential_constraints(valid_graph, valid_explanation_tuples)

            if len(occurential_tuples) == 0:
                continue

            occurential_tuples = choices(occurential_tuples, k=MAX_CONSTRAINTS)
            occurential_tuples = list(set(occurential_tuples))

        else:
            occurential_tuples = []

        if constraint_type == "temporal":
            temporal_tuples = extend_temporal_constraints(valid_graph, valid_explanation_tuples)

            if len(temporal_tuples) == 0:
                continue

            temporal_tuples = choices(temporal_tuples, k=MAX_CONSTRAINTS)
            temporal_tuples = list(set(temporal_tuples))
        
        else:
            temporal_tuples = []


        this_query_result = {
            "query": query,
            "nl_query": query_interpretation + " What is V0?",
            "train_answers": train_answers,
            "train_explanations": train_explanations,
            "train_explanation_tuples": train_explanation_tuples,
            "valid_answers": valid_answers,
            "valid_explanations": valid_explanations,
            "valid_explanation_tuples": valid_explanation_tuples,
            "occurential_tuples": occurential_tuples,
            "temporal_tuples": temporal_tuples
        }

        return this_query_result

        
def sample_general_test(train_graph, valid_graph, test_graph, query_type, _triple2text, constraint_type):

    while True:

        query, _ = pattern_to_query(query_type, test_graph)

        shared={"counter": -1}
        query_interpretation, _, _ = interpretateQueryDirectly(query, shared_var=shared, triple2text=_triple2text)
    
        train_answers, train_explanations, train_explanation_tuples = query_search_answer(train_graph, query)
        valid_answers, valid_explanations, valid_explanation_tuples = query_search_answer(valid_graph, query)
        test_answers, test_explanations, test_explanation_tuples = query_search_answer(test_graph, query)

      
        if len(test_answers) == 0:
            continue


            
        if len(valid_answers) == len(test_answers):
            continue


        if constraint_type == "occurential":

            occurential_tuples = extend_occurential_constraints(test_graph, test_explanation_tuples)
            if len(occurential_tuples) == 0:
                continue

            occurential_tuples = choices(occurential_tuples, k=MAX_CONSTRAINTS)
            occurential_tuples = list(set(occurential_tuples))
        
        else:
            occurential_tuples = []

        if constraint_type == "temporal":

            temporal_tuples = extend_temporal_constraints(test_graph, test_explanation_tuples)

            if len(temporal_tuples) == 0:
                continue
    
        

            temporal_tuples = choices(temporal_tuples, k=MAX_CONSTRAINTS)
            temporal_tuples = list(set(temporal_tuples))
        
        else:
            temporal_tuples = []


        # background_tuples = extend_query(test_graph, query)

        this_query_result = {
            "query": query,
            "nl_query": query_interpretation + " What is V0?",
            "train_answers": train_answers,
            "train_explanations": train_explanations,
            "train_explanation_tuples": train_explanation_tuples,
            "valid_answers": valid_answers,
            "valid_explanations": valid_explanations,
            "valid_explanation_tuples": valid_explanation_tuples,
            "test_answers": test_answers,
            "test_explanations": test_explanations,
            "test_explanation_tuples": test_explanation_tuples,
            "occurential_tuples": occurential_tuples,
            "temporal_tuples": temporal_tuples
        }

        return this_query_result


        



    







if __name__ == "__main__":


    all_query_types = pd.read_csv("./test_generated_formula_anchor_node=3.csv").reset_index(
        drop=True)  # debug
    
    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]
        original_depth = int(all_query_types.original_depth[i])
        if original_depth > 2:
            continue

        if "u" in query or "n" in query:
            continue
        
        

        original_query_types[fid] = query

  



    input_graph_names = [["./aser50k_train.pickle",
    "./aser50k_valid.pickle", "./aser50k_test.pickle"]]


    start_time = time.time()
    for train_graph_name, valid_graph_name, test_graph_name in input_graph_names:
        train_graph = nx.read_gpickle(train_graph_name)
        valid_graph = nx.read_gpickle(valid_graph_name)
        test_graph = nx.read_gpickle(test_graph_name)

        for query_name, query_type in original_query_types.items():
            print(test_graph_name, query_name)
            
         
            number_samples = 1

            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

                this_query_result = sample_general_train(train_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)

                this_query_result = sample_general_train(train_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)
    
            with open("./query_data_dev/" + "aser" + "_train_" + query_name + ".json", "w") as f:
                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")



            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

                this_query_result = sample_general_valid(train_graph, valid_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)

                this_query_result = sample_general_valid(train_graph, valid_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)
            
            with open("./query_data_dev/" + "aser" + "_valid_" + query_name  + ".json", "w") as f:
                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")

            
            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

                this_query_result = sample_general_test(train_graph, valid_graph, test_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)

                this_query_result = sample_general_test(train_graph, valid_graph, test_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)
            

            with open("./query_data_dev/" + "aser" + "_test_" + query_name + ".json", "w") as f:
                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")
            

            print("time elapsed: ", time.time() - start_time)



    

    
