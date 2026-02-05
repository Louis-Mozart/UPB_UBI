import shutil
import os
import random
import time
import itertools
import ast
import numpy as np
import pandas as pd
from typing import Tuple, Set
from collections import defaultdict
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm

# Ontology and ML Frameworks
from rdflib import Graph, Namespace, URIRef, RDF
from owlready2 import get_ontology
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# Ontolearn and Owlapy Imports
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

# Metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score

# Custom helper functions - These should be provided in the repository's 'experiment_helpers.py'
from experiment_helpers import load_model, process_family_df, get_entity_df, get_class_embeddings, pred_wrapper

# --- GLOBAL CONFIGURATION ---
# Use relative paths for GitHub reproducibility
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
KGS_DIR = os.path.join(BASE_PATH, "KGs")
RESULTS_DIR = os.path.join(BASE_PATH, "experimental_results")
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")

# Set pandas options for full output logging
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)


def remove_percentage_of_type(input_owl_path, output_owl_path, type_name, percentage_to_remove):
	"""
    Load an ontology and remove a specific percentage of rdf:type triples for a class.
    """
	g = Graph()
	g.parse(input_owl_path, format='xml')

	# Define namespaces
	FAMILY = Namespace("http://www.benchmark.org/family#")

	# Build type URI
	target_type = URIRef(FAMILY[type_name])

	# Find all individuals of that type
	individuals_of_type = list(g.subjects(RDF.type, target_type))

	# Determine how many to remove
	num_to_remove = int(len(individuals_of_type) * percentage_to_remove)
	individuals_to_remove = random.sample(individuals_of_type, num_to_remove)

	print(
		f"Modifying Knowledge Base: Removing {type_name} type from {num_to_remove}/{len(individuals_of_type)} individuals.")

	# Remove those rdf:type triples
	for ind in individuals_to_remove:
		g.remove((ind, RDF.type, target_type))

	# Save the modified graph
	g.serialize(destination=output_owl_path, format='xml')
	print(f"Modified ontology saved to: {output_owl_path}")


def concept_retrieval(retriever_func, c):
	"""Retrieves string identifiers for individuals belonging to a class expression."""
	return [i.str for i in retriever_func.individuals(c)]


# --- EXPERIMENT PARAMETERS ---
classes_of_interest = [
	"http://www.benchmark.org/family#Brother",
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

# Path to the ground truth Knowledge Base
SYMBOLIC_KB = KnowledgeBase(path=os.path.join(KGS_DIR, "Family", "family-benchmark_rich_background.owl"))

if __name__ == "__main__":
	experiment = "generalization_exp"
	output_path = os.path.join(RESULTS_DIR, experiment, "generalization_results.csv")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	first_write = True

	# Model Checkpoint configuration
	checkpoint_callback = ModelCheckpoint(
		dirpath=os.path.join(CHECKPOINT_DIR, experiment),
		save_top_k=-1,
		save_last=True,
		monitor=None
	)

	datasets = ["Family"]
	removal_percents = range(0, 85, 5)

	# Hyperparameter selection logic
	pqr_list = []
	for p, q, r in itertools.product(range(2), range(2), range(2)):
		value = 1 + p + q + r
		if value != 0 and 32 % value == 0:
			print(f"Selected config: p={p}, q={q}, r={r} (Divisor of 32)")
			pqr_list.append(f"{p}_{q}_{r}")

	results_dict = defaultdict(list)

	for run in range(10):
		random.seed(run + 3)
		for dataset in datasets:
			run_path = os.path.join(KGS_DIR, dataset, experiment, str(run))
			os.makedirs(run_path, exist_ok=True)

			for cl in classes_of_interest:
				class_name = cl.split("#")[-1]

				for removal in removal_percents:
					removal_path = os.path.join(run_path, f"removal_percentage_{removal}")
					os.makedirs(removal_path, exist_ok=True)

					# 1. Create the diminished ontology for this specific class/removal %
					path_diminished = os.path.join(removal_path, f"{dataset.lower()}_modified_{class_name.lower()}.owl")
					remove_percentage_of_type(
						input_owl_path=os.path.join(KGS_DIR, dataset,
						                            f"{dataset.lower()}-benchmark_rich_background.owl"),
						output_owl_path=path_diminished,
						type_name=class_name,
						percentage_to_remove=round(removal / 100, 2)
					)

					# 2. Iterate through reasoning configurations
					for pqr in pqr_list:
						p, q, r = map(int, pqr.split("_"))

						# Initialize Neural Reasoner on the tampered ontology
						neural_owl_reasoner = TripleStoreNeuralReasoner(
							path_of_kb=path_diminished,
							gamma=0.5,
							model='DeCaL',
							p=p, q=q, r=r
						)

						# Anonymized folder path for KGE models
						kge_folder_name = f"KGE_{dataset}_{experiment}_{run}_rem_{removal}_{class_name.lower()}_{pqr}"
						kge_path = os.path.join(BASE_PATH, "temp_storage", kge_folder_name)

						# Load model state
						pqr_model, (_, _) = load_model(path_of_experiment_folder=kge_path)

						# Extract Embeddings
						entity_df = get_entity_df(kge_path, pqr)
						_, _, ind_embeddings, inds = process_family_df(entity_df)
						class_embeddings, classes = get_class_embeddings(entity_df, [cl])

						# Define predictor
						predict_with_class = lambda h: pred_wrapper(h, class_embeddings, model=pqr_model, pqr=pqr,
						                                            base_path=kge_path)

						# 3. Evaluation against Ground Truth (Original KB)
						gt = concept_retrieval(SYMBOLIC_KB, OWLClass(classes[0]))
						gt_set = set(gt)
						labels = np.array([1 if ind in gt_set else 0 for ind in inds])

						# 4. Evaluation against Tampered KB (to identify removed individuals)
						tampered_kb = KnowledgeBase(path=path_diminished)
						tampered_gt = concept_retrieval(tampered_kb, OWLClass(classes[0]))
						tampered_gt_set = set(tampered_gt)
						tampered_labels = np.array([1 if ind in tampered_gt_set else 0 for ind in inds])

						# Identify the test set (individuals that were removed from the class)
						test_set = np.where(labels != tampered_labels)[0]

						# Run Predictions
						results = predict_with_class(ind_embeddings)
						y_preds = np.argmax(results, axis=1)

						# Metrics: All Data
						metrics_all = {
							"recall": recall_score(labels, y_preds),
							"f1": f1_score(labels, y_preds),
							"precision": precision_score(labels, y_preds),
							"accuracy": accuracy_score(labels, y_preds),
							"jaccard": jaccard_score(labels, y_preds)
						}

						# Metrics: Removed Data (The "Generalization" test)
						if len(test_set) == 0:
							tampered_metrics = {k: np.nan for k in metrics_all.keys()}
						else:
							tampered_metrics = {
								"recall": recall_score(labels[test_set], y_preds[test_set]),
								"f1": f1_score(labels[test_set], y_preds[test_set]),
								"precision": precision_score(labels[test_set], y_preds[test_set]),
								"accuracy": accuracy_score(labels[test_set], y_preds[test_set]),
								"jaccard": jaccard_score(labels[test_set], y_preds[test_set])
							}

						# Log results
						row = {
							"run": run, "dataset": dataset, "class_name": class_name, "removal": removal, "pqr": pqr,
							"all_data_recall": metrics_all["recall"],
							"all_data_f1": metrics_all["f1"],
							"all_data_precision": metrics_all["precision"],
							"all_data_accuracy": metrics_all["accuracy"],
							"all_data_jaccard": metrics_all["jaccard"],
							"removed_data_recall": tampered_metrics["recall"],
							"removed_data_f1": tampered_metrics["f1"],
							"removed_data_precision": tampered_metrics["precision"],
							"removed_data_accuracy": tampered_metrics["accuracy"],
							"removed_data_jaccard": tampered_metrics["jaccard"]
						}

						pd.DataFrame([row]).to_csv(output_path, mode="a", header=first_write, index=False)
						first_write = False

						# Cleanup to manage disk usage
						if os.path.exists(kge_path):
							shutil.rmtree(kge_path)