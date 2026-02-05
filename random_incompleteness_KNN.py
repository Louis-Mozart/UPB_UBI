import shutil
import os
import random
import time
import itertools
import ast
from collections import defaultdict
from typing import Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sparse
from rdflib import Graph, Namespace, URIRef, RDF
from owlready2 import get_ontology

# Machine Learning & Metrics
from sklearn.metrics import (
	precision_score, recall_score, accuracy_score, f1_score,
	jaccard_score, pairwise_distances, hamming_loss, zero_one_loss
)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from skmultilearn.adapt import MLkNN
from skmultilearn.utils import get_matrix_in_format

# Ontolearn & Owlapy
from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.triple_store import TripleStore
from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
from owlapy.class_expression import (
	OWLClass, OWLObjectUnionOf, OWLObjectIntersectionOf,
	OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom,
	OWLObjectMinCardinality, OWLObjectMaxCardinality,
	OWLObjectOneOf, OWLObjectComplementOf
)
from owlapy.owl_property import (
	OWLDataProperty, OWLObjectInverseOf, OWLObjectProperty, OWLProperty
)
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl

# Project-specific helper functions (Assume these are in the repo)
from experiment_helpers import load_model, process_family_df, get_entity_df, get_class_embeddings, pred_wrapper

# --- GLOBAL CONFIGURATION ---
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
KGS_DIR = os.path.join(BASE_PATH, "KGs")
RESULTS_DIR = os.path.join(BASE_PATH, "experimental_results")
CHECKPOINTS_DIR = os.path.join(BASE_PATH, "checkpoints")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)


# --- MLkNN PATCHING ---
def _compute_cond(self, X, y):
	"""
    Overridden helper function to compute posterior probabilities for MLkNN.
    Ensures compatibility with specific distance metrics.
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

	cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")
	cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype="float")

	for label in range(self._num_labels):
		for neighbor in range(self.k + 1):
			cond_prob_true[label, neighbor] = (self.s + c[label, neighbor]) / (
					self.s * (self.k + 1) + c_sum[label, 0]
			)
			cond_prob_false[label, neighbor] = (self.s + cn[label, neighbor]) / (
					self.s * (self.k + 1) + cn_sum[label, 0]
			)
	return cond_prob_true, cond_prob_false


# Apply patch
MLkNN._compute_cond = _compute_cond


# --- UTILITY FUNCTIONS ---

def leave_one_out_ml_knn(X, Y, n_neighbors=5, s=0.5):
	"""Compute leave-one-out error for MLkNN using a simplified Bayes rule."""
	nn = MLkNN(k=n_neighbors, s=s)
	nn.fit(X, Y)

	neighs = nn.knn_.kneighbors(X, return_distance=False)
	n_samples, n_labels = Y.shape
	pred = np.zeros_like(Y)

	for i in range(n_samples):
		label_counts = Y[neighs[i]].sum(axis=0)
		for j in range(n_labels):
			count = int(label_counts[j])
			p_true = (nn._cond_prob_true[j, count] * nn._prior_prob_true[j])
			p_false = (nn._cond_prob_false[j, count] * nn._prior_prob_false[j])
			pred[i, j] = 1 if p_true > p_false else 0
	return pred


def make_kb_incomplete_statements(kb_path, output_path, rate):
	"""Removes a percentage of triples (statements) from the KB."""
	kb = get_ontology(kb_path).load()
	all_individuals = list(kb.individuals())

	all_triples = []
	for individual in all_individuals:
		for prop in individual.get_properties():
			for value in prop[individual]:
				all_triples.append((individual, prop, value))

	num_to_remove = int(len(all_triples) * (rate / 100))
	triples_to_remove = random.sample(all_triples, num_to_remove)

	for subject, predicate, obj in triples_to_remove:
		predicate[subject].remove(obj)

	kb.save(file=output_path, format="rdfxml")


def concept_retrieval(retriever_func, c):
	"""Wrapper for Ontolearn individual retrieval."""
	return [i.str for i in retriever_func.individuals(c)]


# --- EXPERIMENT SETUP ---

classes_of_interest = [
	"http://www.benchmark.org/family#Brother", "http://www.benchmark.org/family#Male",
	"http://www.benchmark.org/family#PersonWithASibling", "http://www.benchmark.org/family#Child",
	"http://www.benchmark.org/family#Person", "http://www.benchmark.org/family#Daughter",
	"http://www.benchmark.org/family#Female", "http://www.benchmark.org/family#Father",
	"http://www.benchmark.org/family#Parent", "http://www.benchmark.org/family#Grandchild",
	"http://www.benchmark.org/family#Granddaughter", "http://www.benchmark.org/family#Grandfather",
	"http://www.benchmark.org/family#Grandparent", "http://www.benchmark.org/family#Grandmother",
	"http://www.benchmark.org/family#Grandson", "http://www.benchmark.org/family#Mother",
	"http://www.benchmark.org/family#Sister", "http://www.benchmark.org/family#Son"
]

# Load Ground Truth Knowledge Base
SYMBOLIC_KB_PATH = os.path.join(KGS_DIR, "Family", "family-benchmark_rich_background.owl")
SYMBOLIC_KB = KnowledgeBase(path=SYMBOLIC_KB_PATH)

if __name__ == "__main__":
	experiment = "random_statement_incompleteness_exp"
	datasets = ["Family"]
	removal_percents = range(0, 100, 50)

	# Hyperparameter grids (Adjusted for reproducibility)
	p_s, q_s, r_s = range(1), range(1), range(1)
	pqr_list = [f"{p}_{q}_{r}" for p, q, r in itertools.product(p_s, q_s, r_s)]

	results_dict = defaultdict(list)

	for run in range(1):
		for dataset in datasets:
			run_base = os.path.join(KGS_DIR, dataset, experiment, str(run))
			os.makedirs(run_base, exist_ok=True)

			for removal in removal_percents:
				removal_path = os.path.join(run_base, f"removal_percentage_{removal}")
				os.makedirs(removal_path, exist_ok=True)

				path_diminished = os.path.join(removal_path, f"{dataset.lower()}_modified.owl")

				# 1. Damage Knowledge Base
				make_kb_incomplete_statements(
					kb_path=SYMBOLIC_KB_PATH,
					output_path=path_diminished,
					rate=removal
				)

				# 2. Reasoning and Model Evaluation
				for pqr in pqr_list:
					p, q, r = map(int, pqr.split("_"))

					reasoner = TripleStoreNeuralReasoner(
						path_of_kb=path_diminished,
						gamma=0.5,
						model='DeCaL',
						p=p, q=q, r=r,
						path_to_checkpoint=os.path.join(CHECKPOINTS_DIR, experiment)
					)

					kge_folder = f"KGE_{dataset}_{experiment}_{run}_rem_{removal}_mod_{pqr}"
					kge_path = os.path.join(BASE_PATH, "temp_storage", kge_folder)

					# Load embeddings and model state
					pqr_model, _ = load_model(path_of_experiment_folder=kge_path)
					entity_df = get_entity_df(kge_path, pqr)
					_, _, ind_embeddings, inds = process_family_df(entity_df)
					class_embeddings, classes = get_class_embeddings(entity_df, classes_of_interest)

					multi_labels, multi_preds, multi_tampered_labels = [], [], []

					# 3. Class-specific Prediction Loop
					for i, cl in enumerate(classes):
						predict_with_class = lambda h: pred_wrapper(h, class_embeddings[i], model=pqr_model,
						                                            pqr=pqr, base_path=kge_path)

						# Ground Truth
						gt_set = set(concept_retrieval(SYMBOLIC_KB, OWLClass(classes[i])))
						labels = np.array([1 if ind in gt_set else 0 for ind in inds])

						# Damaged State
						tampered_kb = KnowledgeBase(path=path_diminished)
						tampered_gt_set = set(concept_retrieval(tampered_kb, OWLClass(classes[i])))
						tampered_labels = np.array([1 if ind in tampered_gt_set else 0 for ind in inds])

						# Neural Model Predictions
						y_preds = np.argmax(predict_with_class(ind_embeddings), axis=1)

						multi_labels.append(labels)
						multi_preds.append(y_preds)
						multi_tampered_labels.append(tampered_labels)

					# Stack for Multi-label metrics
					multi_labels = np.stack(multi_labels, axis=1)
					multi_preds = np.stack(multi_preds, axis=1)
					multi_tampered_labels = np.stack(multi_tampered_labels, axis=1)

					# Determine indices of modified triples
					diff_mask = multi_labels != multi_tampered_labels
					test_set = np.where(np.any(diff_mask, axis=1))[0]

					# 4. KNN Distance Metrics (Normalized Euclidean)
					unsup_dist = pairwise_distances(ind_embeddings, metric="euclidean")
					clifford_dists = unsup_dist / unsup_dist.max()

					# 5. MLkNN Evaluation over neighborhoods
					neighborhood_values = range(1, int(0.25 * len(ind_embeddings)))
					for n_val in neighborhood_values:
						knn_preds = leave_one_out_ml_knn(clifford_dists, multi_preds, n_neighbors=n_val)

						# Calculate Global Loss
						h_loss = hamming_loss(multi_labels, knn_preds)
						z_loss = zero_one_loss(multi_labels, knn_preds)

						# Calculate Local Loss (Only on tampered individuals)
						if len(test_set) == 0:
							t_h_loss = t_z_loss = np.nan
						else:
							t_h_loss = hamming_loss(multi_labels[test_set], knn_preds[test_set])
							t_z_loss = zero_one_loss(multi_labels[test_set], knn_preds[test_set])

						# Store results
						results_dict["run"].append(run)
						results_dict["dataset"].append(dataset)
						results_dict["num_neighbors"].append(n_val)
						results_dict["removal"].append(removal)
						results_dict["pqr"].append(pqr)
						results_dict["all_data_hamm_loss"].append(h_loss)
						results_dict["all_data_zero_loss"].append(z_loss)
						results_dict["removed_data_hamm_loss"].append(t_h_loss)
						results_dict["removed_data_zero_loss"].append(t_z_loss)

					# Clean up space
					if os.path.exists(kge_path):
						shutil.rmtree(kge_path)

	# Save aggregated results
	results_df = pd.DataFrame(results_dict)
	output_csv = os.path.join(RESULTS_DIR, experiment, "knn_random_incompleteness_results.csv")
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	results_df.to_csv(output_csv, index=False)
	print(f"✅ Experiment logic complete. Results saved to {output_csv}")