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

from rdflib import Graph, Namespace, URIRef, RDF, RDFS
import os


def get_class_expressions(dataset):
	input_owl_path = f"./KGs/{dataset}/{dataset.lower()}.owl"

	g = Graph()
	g.parse(input_owl_path, format='xml')

	# Namespaces
	FAMILY = Namespace(f"http://www.benchmark.org/{data.lower()}#")
	OWL = Namespace("http://www.w3.org/2002/07/owl#")

	# Collect all class types
	classes = set()

	# Classes explicitly defined as owl:Class
	for cls in g.subjects(RDF.type, OWL.Class):
		classes.add(cls)

	# Sometimes classes are only marked as rdfs:Class, so include those too
	for cls in g.subjects(RDF.type, RDFS.Class):
		classes.add(cls)

	# Print results
	unique_classes = []
	for cls in sorted(classes):
		unique_classes.append(cls)

	return unique_classes

if __name__ == "__main__":
	datasets = []#,"Biopax","Mutagenesis","Carcinogenesis", "Nctrer"]

	p_s = range(2,5)
	q_s = range(2,5)
	r_s = range(2,5)

	pqr_list = []
	for p in p_s:
		for q in q_s:
			for r in r_s:
				value = 1 + p + q + r
				if value != 0 and 32 % value == 0:
					print(f"p={p}, q={q}, r={r} → 1+p+q+r={value} divides 32")
					pqr_list.append(f"{p}_{q}_{r}")

	for data in datasets:

		input_owl_path = f"KGs/{data}/{data.lower()}.owl"

		for pqr in pqr_list:
			p = int(pqr.split("_")[0])
			q = int(pqr.split("_")[1])
			r = int(pqr.split("_")[2])

			# Initialize model for this triple (p, q, r)
			neural_owl_reasoner = TripleStoreNeuralReasoner(
				path_of_kb=input_owl_path,
				gamma=0.5,
				model='DeCaL',
				p=p, q=q, r=r
			)
