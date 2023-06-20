import json
from turtle import forward

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloader
import model

from model import LabelSmoothingLoss, ConstraintFuser


from functools import reduce


from dataloader import TestDataset, ValidDataset, TrainDataset, SingledirectionalOneShotIterator
import pickle
import numpy as np
import math

class MLPMixer(nn.Module):
    def __init__(self, embedding_size, num_patches = 4):
        super(MLPMixer, self).__init__()
    
        self.embedding_size = embedding_size

        self.num_patches = num_patches

        self.mlp1 = MlpOperatorSingle(2 * self.embedding_size // self.num_patches)
        self.mlp2 = MlpOperatorSingle(self.embedding_size)
        
        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.mlp3 = MlpOperator(self.embedding_size)


    def forward(self, x, y):

        # [batch_size, 2, embedding_size]
        input_tensor = torch.stack((x, y), dim=1)

        activations = input_tensor.view(-1, 2, self.num_patches, self.embedding_size // self.num_patches)

        activations = activations.permute(0, 2, 1, 3)

        activations = activations.reshape(-1, self.num_patches, 2 * self.embedding_size // self.num_patches)

        activations = self.mlp1(activations)

        activations = activations.reshape(-1, self.num_patches, 2,  self.embedding_size // self.num_patches)

        activations = activations.permute(0, 2, 1, 3)

        activations = activations.reshape(-1, 2, self.embedding_size)

        activations = activations + input_tensor

        normed_activations = self.layer_norm(activations)

        normed_activations = self.mlp2(normed_activations)

        normed_activations = normed_activations + activations


        return self.mlp3(normed_activations[:,0,:], normed_activations[:,1,:])



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias




class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """

    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class MlpOperator(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()


        self.mlp_layer_1 = nn.Linear(embedding_size * 2, embedding_size)
        self.mlp_layer_2 = nn.Linear(embedding_size, embedding_size // 2)
        self.mlp_layer_3 = nn.Linear(embedding_size //2, embedding_size)

        self.activation = nn.GELU()
    

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = self.activation(self.mlp_layer_1(x))
        x = self.activation(self.mlp_layer_2(x))
        x = self.mlp_layer_3(x)
        return x


class MlpOperatorSingle(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()


        self.mlp_layer_1 = nn.Linear(embedding_size, embedding_size // 2)
        self.mlp_layer_2 = nn.Linear(embedding_size // 2, embedding_size // 2)
        self.mlp_layer_3 = nn.Linear(embedding_size //2, embedding_size)

        self.activation = nn.GELU()
    

    def forward(self, x):
        x = self.activation(self.mlp_layer_1(x))
        x = self.activation(self.mlp_layer_2(x))
        x = self.mlp_layer_3(x)
        return x



class MLPReasoner(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, 
                 dropout_rate=0.3,
             value_vocab=None, ):
        super(MLPReasoner, self).__init__(num_entities, num_relations, embedding_size)


        self.num_entities = num_entities
        self.num_relations = num_relations

    
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
       
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

    
        self.value_vocab = value_vocab

      

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()



        

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

     
        
        
        # MLP operations

        self.projection_mlp = MlpOperator(embedding_size)
        self.union_mlp = MlpOperator(embedding_size)
        self.intersection_mlp = MlpOperator(embedding_size)
        self.negation_mlp = MlpOperatorSingle(embedding_size)


        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights
    
       


    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        query_scores = self.decoder(query_encoding)
        return query_scores

    
    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels).type(torch.LongTensor)
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

        return self.projection_mlp(sub_query_encoding, relation_embeddings)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        
        return  reduce(lambda x, y: self.intersection_mlp(x, y), sub_query_encoding_list)

    
    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        

        return reduce(lambda x, y: self.union_mlp(x, y), sub_query_encoding_list)


    def negation(self, sub_query_encoding):
        """
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        return self.negation_mlp(sub_query_encoding)



class MLPMixerReasoner(model.IterativeModel):
    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, 
                 dropout_rate=0.3,
             value_vocab=None, ):
        super(MLPMixerReasoner, self).__init__(num_entities, num_relations, embedding_size)


        self.num_entities = num_entities
        self.num_relations = num_relations

    
        self.entity_embedding = nn.Embedding(num_entities, embedding_size)
       
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)

    
        self.value_vocab = value_vocab

      

        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()



        

        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)

     
        
        
        # MLP operations

        self.projection_mlp = MLPMixer(embedding_size)
        self.union_mlp = MLPMixer(embedding_size)
        self.intersection_mlp = MLPMixer(embedding_size)
        self.negation_mlp = MlpOperatorSingle(embedding_size)


        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights
    
       


    def scoring(self, query_encoding):
        """

        :param query_encoding: [batch_size, embedding_size]
        :return: [batch_size, num_entities]
        """

        query_scores = self.decoder(query_encoding)
        return query_scores

    
    def loss_fnt(self, query_encoding, labels):
        # The size of the query_encoding is [batch_size, num_particles, embedding_size]
        # and the labels are [batch_size]

        # [batch_size, num_entities]
        query_scores = self.scoring(query_encoding)

        # [batch_size]
        labels = torch.tensor(labels).type(torch.LongTensor)
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

        return self.projection_mlp(sub_query_encoding, relation_embeddings)

    def higher_projection(self, relation_ids, sub_query_encoding):
        return self.projection(relation_ids, sub_query_encoding)


    def intersection(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        
        return  reduce(lambda x, y: self.intersection_mlp(x, y), sub_query_encoding_list)

    
    def union(self, sub_query_encoding_list):
        """

        :param sub_query_encoding_list: a list of the sub-query embeddings of size [batch_size, embedding_size]
        :return:  [batch_size, embedding_size]
        """

        

        return reduce(lambda x, y: self.union_mlp(x, y), sub_query_encoding_list)


    def negation(self, sub_query_encoding):
        """
        :param sub_query_encoding: [batch_size, embedding_size]
        :return: [batch_size, embedding_size]
        """

        return self.negation_mlp(sub_query_encoding)

class SemanticMLPReasoner(MLPReasoner):

    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.1, dropout_rate=0.3, value_vocab=None,
                 eventualities_dict=None, relations_dict=None, id2eventuality=None, id2relation=None):
        super().__init__(num_entities, num_relations, embedding_size, label_smoothing, dropout_rate, value_vocab)

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
        # self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding_weight.requires_grad_(True))

        self.embedding_size = embedding_size

        embedding_weights = self.entity_embedding.weight
        self.decoder = nn.Linear(embedding_size,
                                 num_entities,
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.projection_mlp = MlpOperator(embedding_size)
        self.union_mlp = MlpOperator(embedding_size)
        self.intersection_mlp = MlpOperator(embedding_size)
        self.negation_mlp = MlpOperatorSingle(embedding_size)


class ConstraintMLPReasoner(MLPReasoner):

    def __init__(self, num_entities, num_relations, embedding_size, label_smoothing=0.3, dropout_rate=0.3, value_vocab=None):
        super().__init__(num_entities, num_relations, embedding_size, label_smoothing, dropout_rate, value_vocab)

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
            return self.loss_fnt(this_query_result, label)
           

def test_mlp():
    sample_data_path = "../sampled_data_same/"
    KG_data_path = "../KG_data/"

    train_data_path = sample_data_path + "FB15k-237-betae_train_queries_0.json"
    valid_data_path = sample_data_path + "FB15k-237-betae_valid_queries_0.json"
    test_data_path = sample_data_path + "FB15k-237-betae_test_queries_0.json"
    with open(train_data_path, "r") as fin:
        train_data_dict = json.load(fin)

    with open(valid_data_path, "r") as fin:
        valid_data_dict = json.load(fin)

    with open(test_data_path, "r") as fin:
        test_data_dict = json.load(fin)

    data_path = KG_data_path + "FB15k-237-betae"

    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    gqe_model = ConstraintMLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=384)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()

    batch_size = 5
    train_iterators = {}
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
        batched_query, unified_ids, positive_sample = next(iterator)
        print(batched_query)
        print(unified_ids)
        print(positive_sample)

        query_embedding = gqe_model(batched_query)
        print(query_embedding.shape)
        loss = gqe_model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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
        for batched_query, unified_ids, train_answers, valid_answers in loader:
            print(batched_query)
            print(unified_ids)
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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
        for batched_query, unified_ids, train_answers, valid_answers, test_answers in loader:
            print(batched_query)
            print(unified_ids)

            query_embedding = gqe_model(batched_query)
            result_logs = gqe_model.evaluate_entailment(query_embedding, train_answers)
            print(result_logs)

            result_logs = gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            break


def test_mlp_mixer():
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

    gqe_model = ConstraintMLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=384)
    if torch.cuda.is_available():
        gqe_model = gqe_model.cuda()
    

    semantic_gqe_model = SemanticMLPReasoner(num_entities=nentity, num_relations=nrelation, embedding_size=384,
                                          eventualities_dict=eventualities_dict, relations_dict=relations_dict, id2eventuality=id2eventuality, id2relation=id2relation )

    batch_size = 5
    train_iterators = {}
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

        all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]


        query_embedding = gqe_model(batched_query, all_constraints)
        print(query_embedding.shape)
        loss = gqe_model(batched_query,all_constraints, positive_sample)

        query_embedding = semantic_gqe_model(batched_query)
        print(query_embedding.shape)
        loss = semantic_gqe_model(batched_query, positive_sample)
        print(loss)

    validation_loaders = {}
    for query_type, query_answer_dict in valid_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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

            all_constraints = [ occurential_constraints[i] + temporal_constraints[i] for i in range(len(temporal_constraints))]



            query_embedding = gqe_model(batched_query, all_constraints)

            print("valid_answers")
            print(valid_answers)
            print("train_answers")
            print(train_answers)
      
            result_logs = gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            query_embedding = semantic_gqe_model(batched_query)
            result_logs = semantic_gqe_model.evaluate_generalization(query_embedding, train_answers, valid_answers)
            print(result_logs)

            break

    test_loaders = {}
    for query_type, query_answer_dict in test_data_dict.items():

        if "u" in query_type or "n" in query_type:
            continue

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
           

            result_logs = gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            print(train_answers[0])
            print([len(_) for _ in train_answers])
            print([len(_) for _ in valid_answers])
            print([len(_) for _ in test_answers])

            query_embedding = semantic_gqe_model(batched_query)

            result_logs = semantic_gqe_model.evaluate_generalization(query_embedding, valid_answers, test_answers)
            print(result_logs)

            break


if __name__ == "__main__":
    test_mlp_mixer()