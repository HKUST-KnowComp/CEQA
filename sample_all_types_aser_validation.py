import numpy as np
from collections import defaultdict
import json
import networkx as nx

from graph_matcher import  pattern_to_query, query_search_answer, \
 triple2text_aser, interpretateQueryDirectly, count_edges
import pandas as pd
from tqdm import tqdm


from sample import sample_general_train, sample_general_valid, sample_general_test

from multiprocessing import Pool


num_processes = 28

SMALL = False

def sample_train_data(id):
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
    

    input_graph_names = ["./aser50k_train.pickle"]

    for train_graph_name in input_graph_names:
        train_graph = nx.read_gpickle(train_graph_name)

        for query_name, query_type in original_query_types.items():
            number_of_edges = train_graph.number_of_edges()

            if SMALL:
                number_samples = 3
            
            else:
                number_samples = number_of_edges // num_processes // 2

            print(train_graph_name, query_name, id)
            
            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

                
                this_query_result = sample_general_train(train_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)
                this_query_result = sample_general_train(train_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)

                

            
            
            data_name = "aser"

            with open("./query_data/" + data_name + "_train_" + query_name + "_" + str(id) + ".json", "w") as f:
                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")




def sample_valid_data(id):
    all_query_types = pd.read_csv("./test_generated_formula_anchor_node=3.csv").reset_index(
        drop=True)  # debug
    
    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]

        original_depth = int(all_query_types.original_depth[i])
        if original_depth > 2:
            continue

        # if  "n" in query:
        #     continue

        if "u" in query or "n" in query:
            continue
        # if all_query_types.original_depth[i] > 2:
        #     continue
        original_query_types[fid] = query


    input_graph_names = [["./aser50k_train.pickle",
    "./aser50k_valid.pickle"]]


    for train_graph_name, valid_graph_name in input_graph_names:
        train_graph = nx.read_gpickle(train_graph_name)
        valid_graph = nx.read_gpickle(valid_graph_name)

        for query_name, query_type in original_query_types.items():
            number_of_edges = train_graph.number_of_edges()


            if SMALL:
                number_samples = 3
            
            else:
                number_samples = number_of_edges // num_processes // 20 // 2

            print(valid_graph_name, query_name, id)
            
            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

            

                this_query_result = sample_general_valid(train_graph, valid_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)

                this_query_result = sample_general_valid(train_graph, valid_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)

            
            data_name = "aser"

            with open("./query_data/" + data_name + "_valid_" + query_name + "_" + str(id) + ".json", "w") as f:

                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")
      


def sample_test_data(id):
    all_query_types = pd.read_csv("./test_generated_formula_anchor_node=3.csv").reset_index(
        drop=True)  # debug
    
    original_query_types = {}
    for i in range(all_query_types.shape[0]):
        fid = all_query_types.formula_id[i]
        query = all_query_types.original[i]

        original_depth = int(all_query_types.original_depth[i])
        if original_depth > 2:
            continue

        # if  "n" in query:
        #     continue

        if "u" in query or "n" in query:
            continue
        #  if all_query_types.original_depth[i] > 2:
        #    continue
        original_query_types[fid] = query

    # input_graph_names = [["/home/data/jbai/aser_graph/aser15k_train.pickle",
    # "/home/data/jbai/aser_graph/aser15k_valid.pickle", "/home/data/jbai/aser_graph/aser15k_test.pickle"]]

    input_graph_names = [["./aser50k_train.pickle",
    "./aser50k_valid.pickle", "./aser50k_test.pickle"]]


    for train_graph_name, valid_graph_name, test_graph_name in input_graph_names:
        train_graph = nx.read_gpickle(train_graph_name)
        valid_graph = nx.read_gpickle(valid_graph_name)
        test_graph = nx.read_gpickle(test_graph_name)

        for query_name, query_type in original_query_types.items():
            print(test_graph_name, query_name, id)
            number_of_edges = train_graph.number_of_edges()

            number_of_edges = train_graph.number_of_edges()


            if SMALL:
                number_samples = 3
            
            else:
                number_samples = number_of_edges // num_processes // 20 // 2
            
            this_query_type_list = []
            for _ in tqdm(range(number_samples)):

                this_query_result = sample_general_test(train_graph, valid_graph, test_graph, query_type, triple2text_aser, "occurential")
                this_query_type_list.append(this_query_result)

                this_query_result = sample_general_test(train_graph, valid_graph, test_graph, query_type, triple2text_aser, "temporal")
                this_query_type_list.append(this_query_result)

            
            data_name = "aser"

            with open("./query_data/" + data_name + "_test_" + query_name + "_" + str(id) + ".json", "w") as f:
                for query in this_query_type_list:
                   f.write(json.dumps(query) + "\n")

    



if __name__ == "__main__":

    import time
    start_time = time.time()

    # with Pool(num_processes) as p:
    #     print(p.map(sample_train_data, range(num_processes)))
    

    with Pool(num_processes) as p:
        print(p.map(sample_valid_data, range(num_processes)))

    # with Pool(num_processes) as p:
    #     print(p.map(sample_test_data, range(num_processes)))
