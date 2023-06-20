import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss
import pickle
import numpy as np
# from .dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
# from .model import IterativeModel
import math

from model import ConstraintFuser

class GQE(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False,negative_size=128):
        super(GQE, self).__init__(num_entities, num_relations, embedding_size, use_old_loss)

        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

        self.intersection_nn_layer_1 = nn.Linear(embedding_size, embedding_size // 2)
        self.relu = nn.ReLU()
        self.intersection_nn_layer_2 = nn.Linear(embedding_size // 2, embedding_size)
        self.intersection_transformation_matrix = nn.Linear(embedding_size, embedding_size, bias=False)

        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.use_old_loss = use_old_loss
        self.negative_size = negative_size
    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        # TODO: fix the scoring function here, this function is not correct
        query_scores = self.decoder(query_encoding)
        return query_scores

    def old_loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, embedding_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)
        label_entity_embeddings = self.entity_embedding(labels)

        batch_size = label_entity_embeddings.shape[0]
        # [batch_size]
        random_entity_indices = torch.randint(0, self.num_entities, (self.negative_size,batch_size)).to(
            self.entity_embedding.weight.device)

        # [batch_size, embedding_size]
        negative_samples_embeddings = self.entity_embedding(random_entity_indices)

        query_encoding_norm = torch.sqrt(torch.sum(query_encoding * query_encoding, dim=-1))
        positive_embedding_norm = torch.sqrt(torch.sum(label_entity_embeddings * label_entity_embeddings, dim=-1))
        negative_embedding_norm = torch.sqrt(
            torch.sum(negative_samples_embeddings * negative_samples_embeddings, dim=-1))

        # [batch_size]
        positive_scores = torch.sum(query_encoding * label_entity_embeddings, dim=-1) / \
                          query_encoding_norm / positive_embedding_norm

        # [batch_size]
        negative_scores = torch.sum(query_encoding * negative_samples_embeddings, dim=-1) / \
                          query_encoding_norm / negative_embedding_norm

        relu = nn.ReLU()
        loss = torch.mean(relu(1 + negative_scores - positive_scores))
        
        return loss

    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels)
        labels = labels.to(self.entity_embedding.weight.device)

        _loss = self.label_smoothing_loss(query_scores, labels)

        return _loss

    def projection(self, relation_ids, sub_query_encoding):
        """
        The relational projection of GQE. To fairly evaluate results, we use the same size of relation and use
        TransE embedding structure.

        :param relation_ids: [batch_size]
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        # [batch_size, embedding_size]
        relation_ids = torch.tensor(relation_ids)
        relation_ids = relation_ids.to(self.relation_embedding.weight.device)
        relation_embeddings = self.relation_embedding(relation_ids)

        return relation_embeddings + sub_query_encoding

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)

    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        # [batch_size, number_sub_queries, embedding_size]
        all_subquery_encodings = torch.stack(sub_query_encoding_list, dim=1)

        # [batch_size, number_sub_queries, embedding_size]
        all_subquery_encodings = self.intersection_nn_layer_1(all_subquery_encodings)
        all_subquery_encodings = self.relu(all_subquery_encodings)
        all_subquery_encodings = self.intersection_nn_layer_2(all_subquery_encodings)

        # The implementation of \phi is mean pooling
        # [batch_size, embedding_size]
        all_subquery_encodings = torch.mean(all_subquery_encodings, dim=1)

        # The transformation matrix
        # [batch_size, embedding_size]
        all_subquery_encodings = self.intersection_transformation_matrix(all_subquery_encodings)

        return all_subquery_encodings


class SemanticGQE(GQE):
    """
    The semantic GQE model is incoporated with the semantic information of the eventuality and relation.
    """
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, use_old_loss=False, negative_size=128, 
                 eventualities_dict=None, relations_dict=None, id2eventuality=None, id2relation=None):
        super().__init__(num_entities, num_relations, embedding_size, label_smoothing, use_old_loss, negative_size)
    


        
        evenuality_embedding_list = []
        relation_embedding_list = []

        for i in range(len(id2eventuality)):
            evenuality_embedding_list.append(eventualities_dict[id2eventuality[i]])
        eventuality_embedding_weight = torch.FloatTensor(np.array(evenuality_embedding_list))

        norm = eventuality_embedding_weight.norm(p=2, dim=1, keepdim=True)
        eventuality_embedding_weight = eventuality_embedding_weight.div(norm.expand_as(eventuality_embedding_weight))
        stdv = 1. / math.sqrt(eventuality_embedding_weight.size(1))

        eventuality_embedding_weight = eventuality_embedding_weight * stdv


        self.entity_embedding = nn.Embedding.from_pretrained(eventuality_embedding_weight).requires_grad_(True)

        num_entities, embedding_size = eventuality_embedding_weight.shape

        # for i in range(len(id2relation)):
        #     relation_embedding_list.append(relations_dict[id2relation[i]])
        
        # relation_embedding_weight = torch.FloatTensor(np.array(relation_embedding_list))
        # self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding_weight).requires_grad_(True)

        self.embedding_size = embedding_size

        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights


        self.intersection_nn_layer_1 = nn.Linear(embedding_size, embedding_size // 2)
        self.relu = nn.ReLU()
        self.intersection_nn_layer_2 = nn.Linear(embedding_size // 2, embedding_size)
        self.intersection_transformation_matrix = nn.Linear(embedding_size, embedding_size, bias=False)


class ConstraintGQE(GQE):

    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.3, use_old_loss=False, negative_size=128):
        super().__init__(num_entities, num_relations, embedding_size, label_smoothing, use_old_loss, negative_size)

        
        relation_embedding = self.relation_embedding
        entity_embedding = self.entity_embedding
        self.constraint_fuser = ConstraintFuser(relation_embedding, entity_embedding, num_entities, num_relations, embedding_size)

    
    def forward(self, batched_structured_query, constraint_lists, label=None):

        assert batched_structured_query[0] in ["p", "e", "i"]

        if batched_structured_query[0] == "p":
            
            

            sub_query_result = self.forward(batched_structured_query[2], constraint_lists)
            if batched_structured_query[2][0] == 'e':
                this_query_result = self.projection(batched_structured_query[1], sub_query_result)
            else:
                this_query_result = self.higher_projection(batched_structured_query[1], sub_query_result)
            
            this_query_result = self.constraint_fuser(this_query_result, constraint_lists)

        elif batched_structured_query[0] == "i":
            sub_query_result_list = []
            for _i in range(1, len(batched_structured_query)):
                sub_query_result = self.forward(batched_structured_query[_i], constraint_lists)
                sub_query_result_list.append(sub_query_result)

            this_query_result = self.intersection(sub_query_result_list)

            this_query_result = self.constraint_fuser(this_query_result, constraint_lists)


        elif batched_structured_query[0] == "e":

            entity_ids = torch.tensor(batched_structured_query[1])
            entity_ids = entity_ids.to(self.entity_embedding.weight.device)
            this_query_result = self.entity_embedding(entity_ids)

        else:
            this_query_result = None

        if label is None:
            return this_query_result

        else:
            if self.use_old_loss == False:
                return self.loss_fnt(this_query_result, label)
            else:
                return self.old_loss_fnt(this_query_result, label)
    

    

if __name__ == "__main__":

    sample_data_path = "../query_data_dev_filtered/"
    KG_data_path = "../query_data_dev_filtered/"

    train_data_path = sample_data_path + "query_data_train_filtered.json"
    valid_data_path = sample_data_path + "query_data_valid_filtered.json"
    test_data_path = sample_data_path + "query_data_test_filtered.json"

    train_data_dict = {}
    with open(train_data_path, "r") as fin:
        for line in fin:
            line_dict = json.loads(line.strip())
            query_type = line_dict["query_type"]

            if query_type not in train_data_dict:
                train_data_dict[query_type] = {}
            
            id_query = line_dict["id_query"]
            train_answers = line_dict["id_train_answers"]
            train_data_dict[query_type][id_query] = {"train_answers": train_answers}
            
            train_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
            train_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]

    valid_data_dict = {}
    with open(valid_data_path, "r") as fin:
        for line in fin:
            line_dict = json.loads(line.strip())
            query_type = line_dict["query_type"]

            if query_type not in valid_data_dict:
                valid_data_dict[query_type] = {}
            
            id_query = line_dict["id_query"]
            train_answers = line_dict["id_train_answers"]
            valid_answers = line_dict["id_valid_answers"]

            valid_data_dict[query_type][id_query] = {
                "train_answers": train_answers,
                "valid_answers": valid_answers}


            valid_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
            valid_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]

    test_data_dict = {}
    with open(test_data_path, "r") as fin:
        for line in fin:
            line_dict = json.loads(line.strip())
            query_type = line_dict["query_type"]

            if query_type not in test_data_dict:
                test_data_dict[query_type] = {}
            
            id_query = line_dict["id_query"]
            train_answers = line_dict["id_train_answers"]
            valid_answers = line_dict["id_valid_answers"]
            test_answers = line_dict["id_test_answers"]

            test_data_dict[query_type][id_query] = {
                "train_answers": train_answers,
                "valid_answers": valid_answers,
                "test_answers": test_answers}


            test_data_dict[query_type][id_query]["id_temporal_tuples"] = line_dict["id_temporal_tuples"]
            test_data_dict[query_type][id_query]["id_occurential_tuples"] = line_dict["id_occurential_tuples"]

    
    with open('%sstats.txt' % KG_data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    

    # Load sentences & embeddings from disc
    with open('../_eventuality_embeddings.pkl', "rb") as fIn:
        eventualities_dict = pickle.load(fIn)

    with open('../_relation_embeddings.pkl', "rb") as fIn:
        relations_dict = pickle.load(fIn)



    with open("../query_data_filtered/eventuality2id.json", "r") as f:
        eventuality2id = json.load(f)
        id2eventuality = {v: k for k, v in eventuality2id.items()}

    with open("../query_data_filtered/relation2id.json", "r") as f:
        relation2id = json.load(f)
        id2relation = {v: k for k, v in relation2id.items()}

    gqe_model = ConstraintGQE(num_entities=nentity, num_relations=nrelation, embedding_size=384,use_old_loss=False)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()


    semantic_gqe_model = SemanticGQE(num_entities=nentity, num_relations=nrelation, embedding_size=384,use_old_loss=False, 
                                    eventualities_dict=eventualities_dict, relations_dict=relations_dict, id2eventuality=id2eventuality, id2relation=id2relation )

    batch_size = 5
    train_iterators = {}
    print("train_data_dict.keys()", train_data_dict.keys())
    for query_type, query_answer_dict in train_data_dict.items():


        print("====================================")
        print(query_type)

        new_iterator = dataloader.SingledirectionalOneShotIterator(DataLoader(
            dataloader.TrainDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TrainDataset.collate_fn
        ))
        train_iterators[query_type] = new_iterator

    for key, iterator in train_iterators.items():
        print("read ", key)
        batched_query, unified_ids, positive_sample, is_temporal, is_occurential, temporal_constraints, occurential_constraints = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)


        print("is_temporal", is_temporal)
        print("is_occurential", is_occurential)

        print("temporal_constraints", temporal_constraints)
        print("occurential_constraints", occurential_constraints)
        

        all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]

        query_embedding = gqe_model(batched_query, all_constraints)
        print(query_embedding.shape)
        loss = gqe_model(batched_query, all_constraints, positive_sample)
        loss = semantic_gqe_model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

       
        print("====================================")

        print(query_type)

        new_iterator = DataLoader(
            dataloader.ValidDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.ValidDataset.collate_fn
        )
        validation_loaders[query_type] = new_iterator

    for key, loader in validation_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, is_temporal, is_occurential, temporal_constraints, occurential_constraints in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            print(train_answers)

            print("is_temporal", is_temporal)
            print("is_occurential", is_occurential)

            print("temporal_constraints", temporal_constraints)
            print("occurential_constraints", occurential_constraints)
            
            all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]

            query_embedding = gqe_model(batched_query, all_constraints)
            # result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            # print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            query_embedding = semantic_gqe_model(batched_query)
            # result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            # print(result_logs)

            result_logs = semantic_gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)
            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():


        print("====================================")
        print(query_type)

        new_loader = DataLoader(
            dataloader.TestDataset(nentity, nrelation, query_answer_dict),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataloader.TestDataset.collate_fn
        )
        test_loaders[query_type] = new_loader

    for key, loader in test_loaders.items():
        print("read ", key)
        for batched_query, unified_ids, train_answers, valid_answers, test_answers, is_temporal, is_occurential, temporal_constraints, occurential_constraints in loader:
            print(batched_query)
            print(unified_ids)

            all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]

            query_embedding = gqe_model(batched_query, all_constraints)

            # print(train_answers)
            
            # result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            # print(result_logs)
            print("is_temporal", is_temporal)
            print("is_occurential", is_occurential)

            print("temporal_constraints", temporal_constraints)
            print("occurential_constraints", occurential_constraints)


            result_logs = gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            query_embedding = semantic_gqe_model(batched_query)
            # result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            # print(result_logs)
            result_logs = semantic_gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break
