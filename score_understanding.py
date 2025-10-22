import shutil
from collections import defaultdict

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score

from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
from owlapy.class_expression import (
    OWLClass,
    OWLObjectUnionOf,
    OWLObjectIntersectionOf,
    OWLObjectSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectMinCardinality,
    OWLObjectMaxCardinality,
    OWLObjectOneOf,
    OWLObjectComplementOf
)

from owlapy.owl_property import (
    OWLDataProperty,
    OWLObjectInverseOf,
    OWLObjectProperty,
    OWLProperty,
)
from owlapy.iri import IRI

from owlapy.owl_individual import OWLNamedIndividual
from lightning.pytorch.callbacks import ModelCheckpoint


import time
from typing import Tuple, Set
import pandas as pd
from owlapy import owl_expression_to_dl
from itertools import chain
from argparse import ArgumentParser
import os
from tqdm import tqdm
import random
import itertools
import ast
from owlready2 import get_ontology
import umap
from sklearn.metrics.pairwise import pairwise_distances
from deepview.evaluate import leave_one_out_knn_dist_err
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score

# Set pandas options to ensure full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)

from rdflib import Graph, Namespace, URIRef, RDF
import os
from experiment_helpers import *
classes_of_interest = ["http://www.benchmark.org/family#Brother",
                       "http://www.benchmark.org/family#Male",
                       "http://www.benchmark.org/family#PersonWithASibling",
                       "http://www.benchmark.org/family#Child",
                       "http://www.benchmark.org/family#Person",
                       "http://www.benchmark.org/family#Daughter",
                       "http://www.benchmark.org/family#Female",
                       "http://www.benchmark.org/family#Father",
                       "http://www.benchmark.org/family#Parent",
                       "http://www.benchmark.org/family#Grandchild",
                       "http://www.benchmark.org/family#Granddaughter",
                       "http://www.benchmark.org/family#Grandfather",
                       "http://www.benchmark.org/family#Grandparent",
                       "http://www.benchmark.org/family#Grandmother",
                       "http://www.benchmark.org/family#Grandson",
                       "http://www.benchmark.org/family#Mother",
                       "http://www.benchmark.org/family#Sister",
                       "http://www.benchmark.org/family#Son",
                       ]
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import OWLClass


SYMBOLIC_KB = KnowledgeBase(path="./KGs/Family/family-benchmark_rich_background.owl")

def concept_retrieval(retriever_func, c):
    return [i.str for i in retriever_func.individuals(c)]

base_path = "./UPB_UBI"

if __name__ == "__main__":
    """
    It does not make sense as to why in the paper(https://openreview.net/pdf?id=4qRCiEZGKd) Table 3 contains
    that Jaccard score is 1 for all of the entities but when I run the below code. It gives a different score. 
    We need to understand where this difference is coming from. 
    """


    pqrs = ["0_0_0", "0_0_1", "0_1_0", "0_1_1", "1_0_0", "1_0_1", "1_1_0", "1_1_1"]

    for pqr in pqrs:
        print("PQR", pqr)
        kge_path = f"./KGs_Family_family-benchmark_rich_background_owl_{pqr}"

        # Load model
        pqr_model, (_, _) = load_model(path_of_experiment_folder=kge_path)

        # Get the embeddings of the model
        _, _, ind_embeddings, inds = process_family_df(get_entity_df(kge_path, pqr))

        # Get the class embeddings from the model
        class_embeddings, classes = get_class_embeddings(get_entity_df(kge_path, pqr), classes_of_interest)

        clifford_distance = pairwise_distances(ind_embeddings)

        f1s = []
        recalls = []
        jaccards = []
        LOO_pred_errors = []
        LOO_gt_errors = []
        precisions = []
        for i, cl in enumerate(classes):
            predict_with_class = lambda h: pred_wrapper(h, class_embeddings[i], model=pqr_model, pqr=pqr,
                                                        base_path=kge_path)
            gt = concept_retrieval(SYMBOLIC_KB, OWLClass(classes[i]))
            gt_set = set(gt)
            labels = np.array([1 if ind in gt_set else 0 for ind in inds])
            results = predict_with_class(ind_embeddings)
            y_preds = np.argmax(results, axis=1)

            # umap_embedded = umap.UMAP(metric="precomputed").fit_transform(clifford_distance)

            loo_p = leave_one_out_knn_dist_err(clifford_distance, y_preds, n_neighbors=5)
            loo_gt = leave_one_out_knn_dist_err(clifford_distance, labels, n_neighbors=5)
            LOO_pred_errors.append(loo_p)
            LOO_gt_errors.append(loo_gt)

            precisions.append(precision_score(labels, y_preds))
            recalls.append(recall_score(labels, y_preds))
            f1s.append(f1_score(labels, y_preds))
            jaccards.append(jaccard_score(labels, y_preds))

        print("Average Recall: ", np.mean(recalls))
        print("Average Precision: ", np.mean(precisions))
        print("Average F1: ", np.mean(f1s))
        print("Average Jaccard: ", np.mean(jaccards))
        print("Average QNN Pred Error: ", np.mean(LOO_pred_errors))
        print("Average QNN GT Error: ", np.mean(LOO_gt_errors))
