import argparse
from gqe import GQE, SemanticGQE, ConstraintGQE
from q2p import Q2P, SemanticQ2P, ConstraintQ2P
from mlp import MLPMixerReasoner, MLPReasoner, SemanticMLPReasoner, ConstraintMLPReasoner
from fuzzqe import FuzzQE, ConstraintFuzzQE


import torch
from dataloader import TrainDataset, ValidDataset, TestDataset, SingledirectionalOneShotIterator
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime   
from tensorboardX import SummaryWriter
import gc
import pickle
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import json


def log_aggregation(list_of_logs):
    all_log = {}

    for __log in list_of_logs:
        # Sometimes the number of answers are 0, so we need to remove all the keys with 0 values
        # The average is taken over all queries, instead of over all answers, as is done following previous work. 
        ignore_exd = False
        ignore_ent = False
        ignore_inf = False

        if "exd_num_answers" in __log and __log["exd_num_answers"] == 0:
            ignore_exd = True
        if "ent_num_answers" in __log and __log["ent_num_answers"] == 0:
            ignore_ent = True
        if "inf_num_answers" in __log and __log["inf_num_answers"] == 0:
            ignore_inf = True
            
        
        for __key, __value in __log.items():
            if "num_answers" in __key:
                continue

            else:
                if ignore_ent and "ent_" in __key:
                    continue
                if ignore_exd and "exd_" in __key:
                    continue
                if ignore_inf and "inf_" in __key:
                    continue

                if __key in all_log:
                    all_log[__key].append(__value)
                else:
                    all_log[__key] = [__value]

    average_log = {_key: np.mean(_value) for _key, _value in all_log.items()}

    return average_log


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The training and evaluation script for the models')

    parser.add_argument("--train_query_dir", required=True)
    parser.add_argument("--valid_query_dir", required=True)
    parser.add_argument("--test_query_dir", required=True)
    parser.add_argument('--kg_data_dir', default="KG_data/", help="The path the original kg data")

    parser.add_argument('--log_steps', default=50000, type=int, help='train log every xx steps')
    parser.add_argument('-dn', '--data_name', default="Aser50k", type=str, help='The name of the dataset')
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    parser.add_argument('-d', '--entity_space_dim', default=384, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)

    parser.add_argument("--warm_up_steps", default=1000, type=int)

    parser.add_argument("-m", "--model", required=True)

    parser.add_argument("--checkpoint_path", type=str, default="../logs")
    parser.add_argument("-old", "--old_loss_fnt", action="store_true")
    parser.add_argument("-fol", "--use_full_fol", action="store_true")
    parser.add_argument("-ga", "--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--max_steps", type=int, default=500000)

    parser.add_argument("--few_shot", type=int, default=32)

    args = parser.parse_args()

   
    data_name = "Aser50k"
  
   
    fol_type = "pi" 
    
    loss =  "new-loss"
   
    info = fol_type + "_" + loss 

    with open('../query_data_filtered/stats.txt' ) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = args.checkpoint_path + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name + '/train'
    test_log_dir = args.checkpoint_path  + '/gradient_tape/' + current_time + "_" + args.model + "_" + info + "_" + data_name + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    batch_size = args.batch_size

    test_batch_size = args.batch_size

    evaluating_query_types = []
    evaluating_df = pd.read_csv("../test_generated_formula_anchor_node=3.csv").reset_index(drop=True)

    for i in range(evaluating_df.shape[0]):
        query_structure = evaluating_df.formula_id[i]
        evaluating_query_types.append(query_structure)

    print("Evaluating query types: ", evaluating_query_types)

    training_query_types = []
    training_df = pd.read_csv("../test_generated_formula_anchor_node=2.csv").reset_index(drop=True)

    for i in range(training_df.shape[0]):
        query_structure = training_df.formula_id[i]
        training_query_types.append(query_structure)
    print("Training query types: ", training_query_types)


    # Load sentences & embeddings from disc
    # with open('../_eventuality_embeddings.pkl', "rb") as fIn:
    #     eventualities_dict = pickle.load(fIn)

    # with open('../_relation_embeddings.pkl', "rb") as fIn:
    #     relations_dict = pickle.load(fIn)

    # with open("../query_data_filtered/eventuality2id.json", "r") as f:
    #     eventuality2id = json.load(f)
    #     id2eventuality = {v: k for k, v in eventuality2id.items()}

    # with open("../query_data_filtered/relation2id.json", "r") as f:
    #     relation2id = json.load(f)
    #     id2relation = {v: k for k, v in relation2id.items()}
   

    # create model
    print("====== Initialize Model ======", args.model)
    if args.model == 'gqe':
        model = GQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, use_old_loss=args.old_loss_fnt)
    elif args.model == "q2p":
        model = Q2P(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "mlp":
        model = MLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    
   
    elif args.model == "fuzzqe":
        model = FuzzQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    
    
    elif args.model == 'gqe_con':
        model = ConstraintGQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim, use_old_loss=args.old_loss_fnt)
    elif args.model == "q2p_con":
        model = ConstraintQ2P(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    elif args.model == "mlp_con":
        model = ConstraintMLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)
    
  
    elif args.model == "fuzzqe_con":
        model = ConstraintFuzzQE(num_entities=nentity, num_relations=nrelation, embedding_size=args.entity_space_dim)

    else:
        raise NotImplementedError

    # add scheduler for the transformer model to do warmup, or the model will not converge at all

    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    
    

    if torch.cuda.is_available():
        model = model.cuda()



    
    global_steps = -1

    file_count = -1
    model_name = args.model

    train_data_dict = {}
    with open(args.train_query_dir, "r") as fin:
        for line in fin:
            line_dict = json.loads(line.strip())
            query_type = line_dict["query_type"]

            if query_type not in train_data_dict:
                train_data_dict[query_type] = {}
            
            id_query = line_dict["id_query"]
            train_answers = line_dict["id_train_answers_filtered"]
            train_data_dict[query_type][id_query] = {"train_answers": train_answers}

            train_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
            train_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]
    

    train_iterators = {}
    for query_type, query_answer_dict in train_data_dict.items():
            
            
            

            new_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(nentity, nrelation, query_answer_dict),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=TrainDataset.collate_fn
            ))
            train_iterators[query_type] = new_iterator

    train_iteration_names = list(train_iterators.keys())
    
    total_length = 0
    for key, value in train_iterators.items():
        total_length += value.len

    total_step = total_length * 20


    
    valid_query_file_name = args.valid_query_dir
    valid_data_dict = {}
    with open(valid_query_file_name , "r") as fin:
        for line in fin:
            line_dict = json.loads(line.strip())
            query_type = line_dict["query_type"]

            if query_type not in valid_data_dict:
                valid_data_dict[query_type] = {}
            
            id_query = line_dict["id_query"]
            train_answers = line_dict["id_train_answers_filtered"]
            valid_answers = line_dict["id_valid_answers_filtered"]

            valid_data_dict[query_type][id_query] = {
                "train_answers": train_answers,
                "valid_answers": valid_answers}
            
            
            valid_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
            valid_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]

            
    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        
        new_iterator = DataLoader(
            ValidDataset(nentity, nrelation, query_answer_dict),
            batch_size=test_batch_size,
            shuffle=True,
            collate_fn=ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator
    

    test_data_dict = {}
    for test_query_file_name in tqdm([args.test_query_dir]):
        with open(  test_query_file_name , "r") as fin:
            for line in fin:
                line_dict = json.loads(line.strip())
                query_type = line_dict["query_type"]

                if query_type not in test_data_dict:
                    test_data_dict[query_type] = {}
                
                id_query = line_dict["id_query"]
                train_answers = line_dict["id_train_answers_filtered"]
                valid_answers = line_dict["id_valid_answers_filtered"]
                test_answers = line_dict["id_test_answers_filtered"]

                test_data_dict[query_type][id_query] = {
                    "train_answers": train_answers,
                    "valid_answers": valid_answers,
                    "test_answers": test_answers}
                
                test_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
                test_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]


    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

    

        new_iterator = DataLoader(
            TestDataset(nentity, nrelation, query_answer_dict),
            batch_size=test_batch_size,
            shuffle=True,
            collate_fn=TestDataset.collate_fn
        )
        test_loaders[query_type] = new_iterator



    while True:
   
        print("====== Training ======", model_name, args.train_query_dir)

        if global_steps > args.max_steps:
            break

        
        for step in tqdm(range(total_step)):
            global_steps += 1

            model.train()
            
            
            task_name = np.random.choice(train_iteration_names)
            iterator = train_iterators[task_name]
            batched_query, unified_ids, positive_sample, is_temporal, is_occurential, temporal_constraints, occurential_constraints = next(iterator)
            
            if "con" in args.model:
                all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]
                loss = model(batched_query, all_constraints, positive_sample)
            
            else:
             
                loss = model(batched_query, positive_sample)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (global_steps + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            else: 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            
            if global_steps % 200 == 0:
                train_summary_writer.add_scalar("y-train-" + task_name, loss.item(), global_steps)
            
            save_step = args.log_steps
            model_name = args.model

           
            # Evaluate the model
            if global_steps % args.log_steps == 0:

                # Save the model
                model.eval()
                general_checkpoint_path = args.checkpoint_path + "/" + model_name + "_" + str(global_steps) + "_" + info + "_" + data_name + ".bin"

                
                torch.save({
                    'steps': global_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            

                # Validation
                print("====== Validation ======", model_name)
              
                generalization_58_types_logs = []
                generalization_29_types_logs = []
                generalization_unseen_29_types_logs = []

             
                generalization_58_types_dict = {}

                temporal_29_types_logs = []
                temporal_unseen_29_types_logs = []

                occurential_29_types_logs = []
                occurential_unseen_29_types_logs = []

        
              
                    

                for task_name, loader in tqdm(validation_loaders.items()):

                    all_generalization_logs = []

                    for batched_query, unified_ids, train_answers, valid_answers, is_temporal, is_occurential, temporal_constraints, occurential_constraints in loader:

                        if "con" in args.model:
                            all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]
                            query_embedding = model(batched_query, all_constraints)
                        else:
                            query_embedding = model(batched_query)
                        
                        generalization_logs = model.evaluate_generalization(query_embedding, train_answers, valid_answers)

                        
                        all_generalization_logs.extend(generalization_logs)

                        if task_name in evaluating_query_types:
                            
                            generalization_58_types_logs.extend(generalization_logs)

                        if task_name in training_query_types:
                            generalization_29_types_logs.extend(generalization_logs)

                        if task_name in evaluating_query_types and task_name not in training_query_types:
                            generalization_unseen_29_types_logs.extend(generalization_logs)
                        
                        for _is_temperoal in is_temporal:
                            if _is_temperoal:
                                if task_name in training_query_types:
                                    temporal_29_types_logs.extend(generalization_logs)
                                else:
                                    temporal_unseen_29_types_logs.extend(generalization_logs)
                            
                        for _is_occurential in is_occurential:
                            if _is_occurential:
                                if task_name in training_query_types:
                                    occurential_29_types_logs.extend(generalization_logs)
                                else:
                                    occurential_unseen_29_types_logs.extend(generalization_logs)


                    
                    

                    if task_name not in generalization_58_types_dict:
                        generalization_58_types_dict[task_name] = []
                    generalization_58_types_dict[task_name].extend(all_generalization_logs)

                            
                
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-valid-" + task_name + "-" + key, value, global_steps)
                
                
               
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
                
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
             
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                temporal_29_types_logs = log_aggregation(temporal_29_types_logs)
                temporal_unseen_29_types_logs = log_aggregation(temporal_unseen_29_types_logs)

                occurential_29_types_logs = log_aggregation(occurential_29_types_logs)
                occurential_unseen_29_types_logs = log_aggregation(occurential_unseen_29_types_logs)

                
                for key, value in generalization_58_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-58-types-" + key, value, global_steps)


                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-29-types-" + key, value, global_steps)


                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-unseen-29-types-" + key, value, global_steps)

                for key, value in temporal_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-temporal-29-types-" + key, value, global_steps)
                
                for key, value in temporal_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-temporal-unseen-29-types-" + key, value, global_steps)
                
                for key, value in occurential_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-occurential-29-types-" + key, value, global_steps)
                
                for key, value in occurential_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-valid-occurential-unseen-29-types-" + key, value, global_steps)

                
                print("====== Test ======", model_name)
               
                generalization_58_types_logs = []
                generalization_29_types_logs = []
                generalization_unseen_29_types_logs = []

                temporal_29_types_logs = []
                temporal_unseen_29_types_logs = []

                occurential_29_types_logs = []
                occurential_unseen_29_types_logs = []


                
                generalization_58_types_dict = {}

           
                    

                for task_name, loader in tqdm(test_loaders.items()):

                    
                    all_generalization_logs = []

                    for batched_query, unified_ids, train_answers, valid_answers, test_answers, is_temporal, is_occurential, temporal_constraints, occurential_constraints in loader:

                        if "con" in args.model:
                            all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]
                            query_embedding = model(batched_query, all_constraints)
                        else:  
                            query_embedding = model(batched_query)
                        
                        generalization_logs = model.evaluate_generalization(query_embedding, valid_answers, test_answers)

                        
                        all_generalization_logs.extend(generalization_logs)

                        if task_name in evaluating_query_types:
                            
                            generalization_58_types_logs.extend(generalization_logs)

                        if task_name in training_query_types:
                            
                            generalization_29_types_logs.extend(generalization_logs)

                        if task_name in evaluating_query_types and task_name not in training_query_types:
                            
                            generalization_unseen_29_types_logs.extend(generalization_logs)

                        
                        for _is_temperoal in is_temporal:
                            if _is_temperoal:
                                if task_name in training_query_types:
                                    temporal_29_types_logs.extend(generalization_logs)
                                else:
                                    temporal_unseen_29_types_logs.extend(generalization_logs)
                            
                        for _is_occurential in is_occurential:
                            if _is_occurential:
                                if task_name in training_query_types:
                                    occurential_29_types_logs.extend(generalization_logs)
                                else:
                                    occurential_unseen_29_types_logs.extend(generalization_logs)

                    

                    if task_name not in generalization_58_types_dict:
                        generalization_58_types_dict[task_name] = []
                    generalization_58_types_dict[task_name].extend(all_generalization_logs)
                
                
                
                
                for task_name, logs in generalization_58_types_dict.items():
                    aggregated_generalization_logs = log_aggregation(logs)
                    for key, value in aggregated_generalization_logs.items():
                        test_summary_writer.add_scalar("z-test-" + task_name + "-" + key, value, global_steps)
                

                
              
                generalization_58_types_logs = log_aggregation(generalization_58_types_logs)
               
                generalization_29_types_logs = log_aggregation(generalization_29_types_logs)
             
                generalization_unseen_29_types_logs = log_aggregation(generalization_unseen_29_types_logs)

                temporal_29_types_logs = log_aggregation(temporal_29_types_logs)
                temporal_unseen_29_types_logs = log_aggregation(temporal_unseen_29_types_logs)

                occurential_29_types_logs = log_aggregation(occurential_29_types_logs)
                occurential_unseen_29_types_logs = log_aggregation(occurential_unseen_29_types_logs)


                for key, value in generalization_58_types_logs.items():
                    test_summary_writer.add_scalar("x-test-58-types-" + key, value, global_steps)

              
                for key, value in generalization_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-29-types-" + key, value, global_steps)

                for key, value in generalization_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-unseen-29-types-" + key, value, global_steps)
                
                for key, value in temporal_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-temporal-29-types-" + key, value, global_steps)
                
                for key, value in temporal_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-temporal-unseen-29-types-" + key, value, global_steps)
                
                for key, value in occurential_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-occurential-29-types-" + key, value, global_steps)
                
                for key, value in occurential_unseen_29_types_logs.items():
                    test_summary_writer.add_scalar("x-test-occurential-unseen-29-types-" + key, value, global_steps)

        
                gc.collect()
        

















