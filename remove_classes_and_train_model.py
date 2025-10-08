
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
    

# Example usage:
remove_percentage_of_type(
    input_owl_path="KGs/Family/family-benchmark_rich_background.owl",
    output_owl_path="KGs/Family/modifications/family_modified_grandmother.owl",
    type_name="Grandmother",
    percentage_to_remove= 0.3 # 30%
)



# def remove_assertions(input_owl_path, output_owl_path, assertions_to_remove):
#     # Load the graph
#     g = Graph()
#     g.parse(input_owl_path, format='xml')

#     # Define common namespaces
#     FAMILY = Namespace("http://www.benchmark.org/family#")
#     OWL = Namespace("http://www.w3.org/2002/07/owl#")

#     # Remove the specified assertions
#     for subj_str, pred_str, obj_str in assertions_to_remove:
#         subject = URIRef(FAMILY[subj_str])
        
#         # Handle predicate
#         if pred_str.lower() == 'type':
#             predicate = RDF.type
#         else:
#             predicate = FAMILY[pred_str]

#         # Handle object
#         if pred_str.lower() == 'type':
#             if obj_str == "NamedIndividual":
#                 obj = URIRef(OWL.NamedIndividual)
#             else:
#                 obj = URIRef(FAMILY[obj_str])
#         else:
#             obj = URIRef(FAMILY[obj_str])

#         # Remove the triple
#         g.remove((subject, predicate, obj))

#     # Serialize to a new file
#     g.serialize(destination=output_owl_path, format='xml')
#     print(f"Updated ontology saved to: {output_owl_path}")

# Example usage
assertions_to_remove = [
    ("F2F28", "type", "Grandmother"),
    ("F10F172", "type", "Grandmother"),
    ("F10F186", "type", "Grandmother"),
    ("F10F195", "type", "Grandmother"),
    ("F2F10", "type", "Grandmother"),
    ("F2F12", "type", "Grandmother"),
    ("F2F19", "type", "Grandmother"),
    ("F2F22", "type", "Grandmother"),
    ("F2F30", "type", "Grandmother"),
    ("F3F41", "type", "Grandmother"),
    ("F3F42", "type", "Grandmother"),
     ("F3F46", "type", "Grandmother"),
    # Add more if needed: ("F2F28", "hasChild", "F2F30")
]

# remove_assertions(
#     input_owl_path="KGs/Family/family.owl",
#     output_owl_path="KGs/Family/family_modified.owl",
#     assertions_to_remove=assertions_to_remove
# )





def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
    start_time = time.time()
    return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

# removed_triples = {(F10F172,type, Grandmother), (F10F186, type, Grandmother), F10F195, F2F10, F2F12, F2F19, F2F22, F2F28}

path_true = "KGs/Family/family-benchmark_rich_background.owl"
path_diminished = "KGs/Family/modifications/family_modified_grandmother.owl"

onto = get_ontology(path_true).load()
named_individuals = {ind.iri for ind in onto.individuals()}


symbolic_kb_true = KnowledgeBase(path=path_true) # The True Knowledge base
symbolic_kb_diminished = KnowledgeBase(path=path_diminished) # The Knowledge base without the omited triple


neural_owl_reasoner = TripleStoreNeuralReasoner(path_of_kb=path_diminished, gamma=0.5)#(path_of_kb=path_diminished, gamma=0.081) This is to play with the removed (stefan,type,father)


# for i in range(len(assertions_to_remove)):
#     print(f"Individual: {assertions_to_remove[i][0]}")
#     print(neural_owl_reasoner.model.predict(h="http://www.benchmark.org/family#"+assertions_to_remove[i][0], r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://www.benchmark.org/family#Grandmother", logits=False))



expression = OWLClass("http://www.benchmark.org/family#Grandmother")



# print(owl_expression_to_dl(expression))


retrieval_y, runtime_y = concept_retrieval(symbolic_kb_true, expression) #The groundtruth retrieval
# () Retrieve a set of inferred individuals and elapsed runtime.
retrieval_ebr, runtime_ebr = concept_retrieval(neural_owl_reasoner, expression) #Retrieval for EBR 


retrieval_ebr = {iri for iri in retrieval_ebr if iri in named_individuals}


retrieval_fic, runtime_fic = concept_retrieval(symbolic_kb_diminished, expression) #Retrieval for EBR 


# () Compute the Jaccard similarity.
jaccard_sim_EBR = jaccard_similarity(retrieval_y, retrieval_ebr)
jaccard_sim_fic = jaccard_similarity(retrieval_y, retrieval_fic)


print(f"Jaccard similarity EBR: {jaccard_sim_EBR}")
print(f"Jaccard similarity FIC: {jaccard_sim_fic}")

print(f"retrieved EBR: {len(retrieval_ebr)}")
print(f"retrieved FIC: {retrieval_fic}")



