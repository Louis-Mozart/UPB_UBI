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

# Set pandas options to ensure full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)

from rdflib import Graph, Namespace, URIRef, RDF
import os
from experiment_helpers import *

from skmultilearn.adapt import MLkNN
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
from skmultilearn.utils import get_matrix_in_format

import numpy as np
# from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss


def _compute_cond(self, X, y):
    """Helper function to compute for the posterior probabilities

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse
        input features, can be a dense or sparse matrix of size
        :code:`(n_samples, n_features)`
    y : numpy.ndaarray or scipy.sparse {0,1}
        binary indicator matrix with label assignments.

    Returns
    -------
    numpy.ndarray
        the posterior probability given true
    numpy.ndarray
        the posterior probability given false
    """

    self.knn_ = NearestNeighbors(n_neighbors=self.k, metric="precomputed").fit(X)
    c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")
    cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="i8")

    label_info = get_matrix_in_format(y, "dok")

    neighbors = [
        a[self.ignore_first_neighbours:]
        for a in self.knn_.kneighbors(
            X, self.k + self.ignore_first_neighbours, return_distance=False
        )
    ]

    for instance in range(self._num_instances):
        deltas = label_info[neighbors[instance], :].sum(axis=0)
        for label in range(self._num_labels):
            if label_info[instance, label] == 1:
                c[label, deltas[0, label]] += 1
            else:
                cn[label, deltas[0, label]] += 1

    c_sum = c.sum(axis=1)
    cn_sum = cn.sum(axis=1)

    cond_prob_true = sparse.lil_matrix(
        (self._num_labels, self.k + 1), dtype="float"
    )
    cond_prob_false = sparse.lil_matrix(
        (self._num_labels, self.k + 1), dtype="float"
    )
    for label in range(self._num_labels):
        for neighbor in range(self.k + 1):
            cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
                    self.s * (self.k + 1) + c_sum[label, 0]
            )
            cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
                    self.s * (self.k + 1) + cn_sum[label, 0]
            )
    return cond_prob_true, cond_prob_false


# replace the method on the class
MLkNN._compute_cond = _compute_cond




def leave_one_out_ml_knn_dist_err(X, Y, n_neighbors=5, s=0.5):
    """
    Compute leave-one-out error for MLkNN.

    X: (n_samples, n_features) feature matrix
    Y: (n_samples, n_labels) binary indicator matrix of labels
    """
    nn = MLkNN(k=n_neighbors, s=s)
    nn.fit(X, Y)

    # get neighbors for all points (including self)
    neighs = nn.knn_.kneighbors(X, return_distance=False)

    n_samples, n_labels = Y.shape
    pred = np.zeros_like(Y)

    for i in range(n_samples):
        # sum neighbor labels
        label_counts = Y[neighs[i]].sum(axis=0)

        # apply MLkNN decision rule: pick label if posterior for "present" > "absent"
        for j in range(n_labels):
            # compute posteriors (simplified Bayes rule as in MLkNN paper)
            count = int(label_counts[j])
            p_true = (nn._cond_prob_true[j, count] * nn._prior_prob_true[j])
            p_false = (nn._cond_prob_false[j, count] * nn._prior_prob_false[j])
            pred[i, j] = 1 if p_true > p_false else 0

    # Hamming loss = fraction of misclassified labels
    return hamming_loss(Y, pred), zero_one_loss(Y, pred)

def remove_percentage_of_type(input_owl_path, output_owl_path, type_name, percentage_to_remove):
    # Load the ontology
    g = Graph()
    g.parse(input_owl_path, format='xml')

    # Define namespaces
    FAMILY = Namespace("http://www.benchmark.org/family#")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")

    # Build type URI
    target_type = URIRef(FAMILY[type_name])

    # Find all individuals of that type
    individuals_of_type = list(g.subjects(RDF.type, target_type))

    # Determine how many to remove
    num_to_remove = int(len(individuals_of_type) * percentage_to_remove)
    individuals_to_remove = random.sample(individuals_of_type, num_to_remove)

    print(f"Removing rdf:type {type_name} from {num_to_remove} out of {len(individuals_of_type)} individuals.")

    # Remove those rdf:type triples
    for ind in individuals_to_remove:
        g.remove((ind, RDF.type, target_type))

    # Serialize the updated graph
    g.serialize(destination=output_owl_path, format='xml')
    print(f"Modified ontology saved to: {output_owl_path}")


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


if __name__ == "__main__":
    # parameters
    experiment = "generalization_exp"
    datasets = ["Family"]  # os.listdir("/home/iroberts/projects/UPB_UBI/LPs")
    removal_percents = range(0, 100,5)
    p_s = range(10)
    q_s = range(10)
    r_s = range(10)

    # Prepare a dictionary to store results
    results_dict = defaultdict(list)
    for run in range(5):
        for dataset in datasets:
            for cl in classes_of_interest:
                # gets name of class → "Brother"
                class_name = cl.split("#")[-1]

                for removal in removal_percents:

                    # We remove a percentage from a class
                    # and keep the rest of the individuals the same
                    remove_percentage_of_type(
                        input_owl_path=f"./KGs/{dataset}/{dataset.lower()}-benchmark_rich_background.owl",
                        output_owl_path=f"./KGs/{dataset}/{experiment}/{dataset.lower()}_{removal}_modified_{class_name.lower()}.owl",
                        type_name=class_name,
                        percentage_to_remove=round(removal / 100, 2)
                    )

                    # Train models on the ontology with the partially removed class
                    path_diminished = f"KGs/{dataset}/{experiment}/{dataset.lower()}_{removal}_modified_{class_name.lower()}.owl"

                    for p in p_s:
                        for q in q_s:
                            for r in r_s:

                                # Initialize model for this triple (p, q, r)
                                neural_owl_reasoner = TripleStoreNeuralReasoner(
                                    path_of_kb=path_diminished,
                                    gamma=0.5,
                                    model='DeCaL',
                                    p=p, q=q, r=r
                                )

                                pqr = f"{p}_{q}_{r}"
                                kge_path = f"./KGs_{dataset}_{experiment}_{dataset.lower()}_{removal}_modified_{class_name.lower()}_owl_{pqr}"

                                # Load model
                                pqr_model, (_, _) = load_model(path_of_experiment_folder=kge_path)

                                # Get the embeddings of the model
                                _, _, ind_embeddings, inds = process_family_df(get_entity_df(kge_path, pqr))

                                # Get the class embeddings from the model
                                class_embeddings, classes = get_class_embeddings(get_entity_df(kge_path, pqr), [cl])

                                # Create predictor for the class in question
                                predict_with_class = lambda h: pred_wrapper(h, class_embeddings, model=pqr_model, pqr=pqr,base_path=kge_path)

                                # Get ground truth labels
                                gt = concept_retrieval(symbolic_kb, OWLClass(classes[0]))
                                gt_set = set(gt)

                                # Give 1 if part of the class, 0 otherwise
                                labels = np.array([1 if ind in gt_set else 0 for ind in inds])

                                # Same as above except for with the removed class
                                tampered_kb = KnowledgeBase(path=path_diminished)
                                tampered_gt = concept_retrieval(tampered_kb, OWLClass(classes[0]))
                                tampered_gt_set = set(tampered_gt)
                                tampered_labels = np.array([1 if ind in tampered_gt_set else 0 for ind in inds])

                                # Discover the position of which individuals were removed
                                test_set = np.where(labels != tampered_labels)[0]

                                # Predictions
                                results = predict_with_class(ind_embeddings)
                                y_preds = np.argmax(results, axis=1)

                                # Metrics for all data
                                recall_val = recall_score(labels, y_preds)
                                f1_val = f1_score(labels, y_preds)
                                precision_val = precision_score(labels, y_preds)
                                accuracy_val = accuracy_score(labels, y_preds)
                                jaccard_val = jaccard_score(labels, y_preds)

                                # Metrics for removed individuals only
                                if len(test_set) == 0:
                                    # If no removed individuals, fill with NaNs
                                    tampered_recall = tampered_f1 = tampered_precision = tampered_accuracy = tampered_jaccard = np.nan
                                else:
                                    tampered_recall = recall_score(labels[test_set], y_preds[test_set])
                                    tampered_f1 = f1_score(labels[test_set], y_preds[test_set])
                                    tampered_precision = precision_score(labels[test_set], y_preds[test_set])
                                    tampered_accuracy = accuracy_score(labels[test_set], y_preds[test_set])
                                    tampered_jaccard = jaccard_score(labels[test_set], y_preds[test_set])

                                # Store results (always, even if NaNs)
                                results_dict["run"].append(run)
                                results_dict["dataset"].append(dataset)
                                results_dict["class_name"].append(class_name)
                                results_dict["removal"].append(removal)
                                results_dict["pqr"].append(pqr)
                                results_dict["all_data_recall"].append(recall_val)
                                results_dict["all_data_f1"].append(f1_val)
                                results_dict["all_data_precision"].append(precision_val)
                                results_dict["all_data_accuracy"].append(accuracy_val)
                                results_dict["all_data_jaccard"].append(jaccard_val)
                                results_dict["removed_data_recall"].append(tampered_recall)
                                results_dict["removed_data_f1"].append(tampered_f1)
                                results_dict["removed_data_precision"].append(tampered_precision)
                                results_dict["removed_data_accuracy"].append(tampered_accuracy)
                                results_dict["removed_data_jaccard"].append(tampered_jaccard)

                                # Delete kge_path folder to save space
                                if os.path.exists(kge_path):
                                    shutil.rmtree(kge_path)

    # After all loops, create DataFrame once and save
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("generalization_results.csv", index=False)
    print(f"✅ Results saved to generalization_results.csv ({len(results_df)} rows)")



