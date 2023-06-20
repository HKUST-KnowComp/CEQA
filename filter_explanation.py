from verifier import verify_discourse, verify_temporal

from graph_matcher import extend_occurential_constraints

from z3 import Bool, solve, Implies, Not, Or, Solver, And, Real
import json
from tqdm import tqdm
import os

import networkx as nx
import numpy as np

from multiprocessing import Pool

num_processes = 28


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




def filter_train_answers(sample):

    explanation_tuples = sample["train_explanation_tuples"]

    occurential_constraints = sample["occurential_tuples"]
    termporal_constraints = sample["temporal_tuples"]

    train_answers = sample["train_answers"]

    exp_tuples = [e + occurential_constraints + termporal_constraints for e in explanation_tuples]

    filtered_answers = []
    filtered_explanations = []

    for answer_id, explanation_tuple in enumerate(exp_tuples):
        isLogical = verify_discourse(explanation_tuple)
        isTemporalLogical = verify_temporal(explanation_tuple)

        if isLogical != -1 and isTemporalLogical != -1:
            filtered_answers.append(train_answers[answer_id])
            filtered_explanations.append(explanation_tuple)
    
    sample["train_answers_filtered"] = filtered_answers
    sample["train_explanation_tuples_filtered"] = filtered_explanations

    return sample

def filter_validation_answers(sample):

    explanation_tuples = sample["valid_explanation_tuples"]

    occurential_constraints = sample["occurential_tuples"]
    termporal_constraints = sample["temporal_tuples"]

    validation_answers = sample["valid_answers"]

    exp_tuples = [e + occurential_constraints + termporal_constraints for e in explanation_tuples]


    filtered_answers = []
    filtered_explanations = []

    for answer_id, explanation_tuple in enumerate(exp_tuples):
        isLogical = verify_discourse(explanation_tuple)
        isTemporalLogical = verify_temporal(explanation_tuple)

        if isLogical != -1 and isTemporalLogical != -1:
            filtered_answers.append(validation_answers[answer_id])
            filtered_explanations.append(explanation_tuple)
    
    sample["valid_answers_filtered"] = filtered_answers
    sample["valid_explanation_tuples_filtered"] = filtered_explanations

    return sample

def filter_test_answers(sample):
    
    explanation_tuples = sample["test_explanation_tuples"]

    occurential_constraints = sample["occurential_tuples"]
    termporal_constraints = sample["temporal_tuples"]

    test_answers = sample["test_answers"]

    exp_tuples = [e + occurential_constraints + termporal_constraints for e in explanation_tuples]

   

    filtered_answers = []
    filtered_explanations = []

    for answer_id, explanation_tuple in enumerate(exp_tuples):
        isLogical = verify_discourse(explanation_tuple)
        isTemporalLogical = verify_temporal(explanation_tuple)

        if isLogical != -1 and isTemporalLogical != -1:
            filtered_answers.append(test_answers[answer_id])
            filtered_explanations.append(explanation_tuple)
    
    sample["test_answers_filtered"] = filtered_answers
    sample["test_explanation_tuples_filtered"] = filtered_explanations

    return sample




    

if __name__ == "__main__":


    input_graph_names = [["/home/data/jbai/aser_graph/aser50k_train.pickle",
    "/home/data/jbai/aser_graph/aser50k_valid.pickle", "/home/data/jbai/aser_graph/aser50k_test.pickle"]]

    train_graph_name, valid_graph_name, test_graph_name = input_graph_names[0]

    train_graph = nx.read_gpickle(train_graph_name)
    valid_graph = nx.read_gpickle(valid_graph_name)
    test_graph = nx.read_gpickle(test_graph_name)

    eventuality2id = {}
    for node in test_graph.nodes:
        if node not in eventuality2id:
            eventuality2id[node] = len(eventuality2id)

            
    relation2id = {}
    for head, tail, relation_dict in test_graph.edges(data=True):
        for key in relation_dict.keys():
            if key not in relation2id:
                relation2id[key] = len(relation2id)

    def query2id(query_with_nl):

        # print(query_with_nl)

        query_with_nl = query_with_nl[1:-1]
        parenthesis_count = 0

        sub_queries = []
        jj = 0

        
        for ii, character in enumerate(query_with_nl):
            # Skip the comma inside a parenthesis
            if character == "(":
                parenthesis_count += 1

            elif character == ")":
                parenthesis_count -= 1

            if parenthesis_count > 0:
                continue

            if character == ",":
                sub_queries.append(query_with_nl[jj: ii])
                jj = ii + 1

        sub_queries.append(query_with_nl[jj: len(query_with_nl)])

        # print("sub_query: ", sub_queries)
        

        if sub_queries[0] == "p":
            converted_sub_queries = query2id(sub_queries[2])
            relation_id = relation2id[sub_queries[1][1:-1]]

            converted = "(p,({}),{})".format(relation_id, converted_sub_queries)
            # print("converted: ", converted)
            return converted


        elif sub_queries[0] == "e":
            converted = "(e,({}))".format(eventuality2id[sub_queries[1][1:-1]])
            # print("converted: ", converted)
            return converted
        

        elif sub_queries[0] == "i":

            converted_sub_queries = []
            for i in range(1, len(sub_queries)):
                converted_subquery = query2id(sub_queries[i])
                converted_sub_queries.append(converted_subquery)
            
            concatenated_sub_queries = ",".join(converted_sub_queries)

            converted = "(i,{})".format(concatenated_sub_queries)
            # print("converted: ", converted)
            return converted

        else:
            print("Invalid Pattern")
            print(sub_queries)
            exit()




    all_input_train_queries = []
    all_input_validation_queries = []
    all_input_test_queries = []

    directory = "./query_data/"

    for filename in tqdm(os.listdir(directory)):

        query_type = filename.split("_")[-2]
       


        with open(directory + filename, "r") as fin:

            
            for line in fin:
                line_dict = json.loads(line.strip())
                line_dict["query_type"] = query_type
                if "train" in filename:
                    all_input_train_queries.append(line_dict)
                elif "valid" in filename:
                    all_input_validation_queries.append(line_dict)
                else:
                    all_input_test_queries.append(line_dict)

    print("Total train queries: ", len(all_input_train_queries))
    print("Total validation queries: ", len(all_input_validation_queries))
    print("Total test queries: ", len(all_input_test_queries))

    print("Verification starts")




    with Pool(num_processes) as p:
        print("Start processing train queries")
        result_train_queries = list(tqdm(p.imap(filter_train_answers, all_input_train_queries), total=len(all_input_train_queries)))
        print("Start processing validation queries")
        result_validation_queries = list(tqdm(p.imap(filter_train_answers, all_input_validation_queries), total=len(all_input_validation_queries)))
        result_validation_queries = list(tqdm(p.imap(filter_validation_answers, result_validation_queries), total=len(result_validation_queries)))
        print("Start processing test queries")
        result_test_queries = list(tqdm(p.imap(filter_train_answers, all_input_test_queries), total=len(all_input_test_queries)))
        result_test_queries = list(tqdm(p.imap(filter_validation_answers, result_test_queries), total=len(result_test_queries)))
        result_test_queries = list(tqdm(p.imap(filter_test_answers, result_test_queries), total=len(result_test_queries)))
    print("Verification ends")

    print("Total train queries: ", len(result_train_queries))
    print("Total validation queries: ", len(result_validation_queries))
    print("Total test queries: ", len(result_test_queries))

    print("Removing empty and trivial queries")
    result_train_queries = [q for q in result_train_queries if len(q["train_answers_filtered"]) > 0 and len(q["train_answers_filtered"]) != len(q["train_answers"])]
    result_validation_queries = [q for q in result_validation_queries if len(q["valid_answers_filtered"]) > 0 and len(q["valid_answers_filtered"]) != len(q["valid_answers"])]
    result_test_queries = [q for q in result_test_queries if len(q["test_answers_filtered"]) > 0 and len(q["test_answers_filtered"]) != len(q["test_answers"])]
    print("Total train queries: ", len(result_train_queries))
    print("Total validation queries: ", len(result_validation_queries))
    print("Total test queries: ", len(result_test_queries))


    train_answer_length = []
    train_filtered_answer_length = []
    valid_answer_length = []
    valid_filtered_answer_length = []
    test_answer_length = []
    test_filtered_answer_length = []
    for query in result_test_queries:
        test_answer_length.append(len(query["test_answers"]))
        test_filtered_answer_length.append(len(query["test_answers_filtered"]))
        valid_answer_length.append(len(query["valid_answers"]))
        valid_filtered_answer_length.append(len(query["valid_answers_filtered"]))
        train_answer_length.append(len(query["train_answers"]))
        train_filtered_answer_length.append(len(query["train_answers_filtered"]))
    
    print("Train answer length: ", np.mean(train_answer_length))
    print("Train filtered answer length: ", np.mean(train_filtered_answer_length))
    print("Valid answer length: ", np.mean(valid_answer_length))
    print("Valid filtered answer length: ", np.mean(valid_filtered_answer_length))
    print("Test answer length: ", np.mean(test_answer_length))
    print("Test filtered answer length: ", np.mean(test_filtered_answer_length))


    print("Start converting to ids")
    o_have_anchor_count = 0
    o_no_anchor_count = 0
    t_have_anchor_count = 0
    t_no_anchor_count = 0
    for query_sample in tqdm(result_train_queries):
        query_type = query_sample["query_type"]

        query_in_nl = query_sample["query"]

        query_in_id = query2id(query_in_nl)
        train_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers"]]
        train_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers_filtered"]]


        query_sample["id_query"] = query_in_id
        query_sample["id_train_answers"] = train_answer_in_id
        query_sample["id_train_answers_filtered"] = train_filtered_answer_in_id

        query_sample["id_occurential_tuples"] =  [(eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]) for t in query_sample["occurential_tuples"]] 
        query_sample["id_temporal_tuples"] =  [(eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]) for t in query_sample["temporal_tuples"]]

        
       

        anchors = get_anchor_nodes(query_sample["query"])
    

        
        for constraint_tuple in query_sample["occurential_tuples"]:
            if constraint_tuple[0] in anchors or constraint_tuple[1] in anchors:
                o_have_anchor_count += 1
            else:
                o_no_anchor_count += 1

        
        for constraint_tuple in query_sample["temporal_tuples"]:
            if constraint_tuple[0] in anchors or constraint_tuple[1] in anchors:
                t_have_anchor_count += 1
            else:
                t_no_anchor_count += 1
        
    print("Temporal Have anchor count: ", t_have_anchor_count)
    print("Temporal No anchor count: ", t_no_anchor_count)
    print("Occurence Have anchor count: ", o_have_anchor_count)
    print("Occurence No anchor count: ", o_no_anchor_count)



    for query_sample in tqdm(result_validation_queries):
        query_type = query_sample["query_type"]

        query_in_nl = query_sample["query"]

        query_in_id = query2id(query_in_nl)
        rain_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers"]]
        train_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers_filtered"]]
        valid_answer_in_id = [eventuality2id[a] for a in query_sample["valid_answers"]]
        valid_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["valid_answers_filtered"]]


        query_sample["id_query"] = query_in_id
        query_sample["id_train_answers"] = train_answer_in_id
        query_sample["id_train_answers_filtered"] = train_filtered_answer_in_id

        query_sample["id_valid_answers"] = valid_answer_in_id
        query_sample["id_valid_answers_filtered"] = valid_filtered_answer_in_id

        query_sample["id_occurential_tuples"] =  [[eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]] for t in query_sample["occurential_tuples"]] 
        query_sample["id_temporal_tuples"] =  [[eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]] for t in query_sample["temporal_tuples"]]

    
    for query_sample in tqdm(result_test_queries):
        query_type = query_sample["query_type"]

        query_in_nl = query_sample["query"]

        query_in_id = query2id(query_in_nl)
        train_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers"]]
        train_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["train_answers_filtered"]]
        valid_answer_in_id = [eventuality2id[a] for a in query_sample["valid_answers"]]
        valid_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["valid_answers_filtered"]]
        test_answer_in_id = [eventuality2id[a] for a in query_sample["test_answers"]]
        test_filtered_answer_in_id = [eventuality2id[a] for a in query_sample["test_answers_filtered"]]


        query_sample["id_query"] = query_in_id
        query_sample["id_train_answers"] = train_answer_in_id
        query_sample["id_train_answers_filtered"] = train_filtered_answer_in_id

        query_sample["id_valid_answers"] = valid_answer_in_id
        query_sample["id_valid_answers_filtered"] = valid_filtered_answer_in_id

        query_sample["id_test_answers"] = test_answer_in_id
        query_sample["id_test_answers_filtered"] = test_filtered_answer_in_id

        query_sample["id_occurential_tuples"] =  [[eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]] for t in query_sample["occurential_tuples"]] 
        query_sample["id_temporal_tuples"] =  [[eventuality2id[t[0]], eventuality2id[t[1]], relation2id[t[2]]] for t in query_sample["temporal_tuples"]]


    print("Start saving")
    with open("./query_data_filtered/query_data_train_filtered.json", "w") as fout:
        for query in result_train_queries:
            fout.write(json.dumps(query) + "\n")
    
    with open("./query_data_filtered/query_data_valid_filtered.json", "w") as fout:
        for query in result_validation_queries:
            fout.write(json.dumps(query) + "\n")
    
    with open("./query_data_filtered/query_data_test_filtered.json", "w") as fout:
        for query in result_test_queries:
            fout.write(json.dumps(query) + "\n")

        
      


               
                

            

