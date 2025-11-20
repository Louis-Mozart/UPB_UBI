
# from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner
# from ontolearn.knowledge_base import KnowledgeBase
# from ontolearn.triple_store import TripleStore
# from ontolearn.utils import jaccard_similarity, f1_set_similarity, concept_reducer, concept_reducer_properties
# from owlapy.class_expression import (
#     OWLClass,
#     OWLObjectUnionOf,
#     OWLObjectIntersectionOf,
#     OWLObjectSomeValuesFrom,
#     OWLObjectAllValuesFrom,
#     OWLObjectMinCardinality,
#     OWLObjectMaxCardinality,
#     OWLObjectOneOf,
#     OWLObjectComplementOf
# )


# from owlapy.owl_property import (
#     OWLDataProperty,
#     OWLObjectInverseOf,
#     OWLObjectProperty,
#     OWLProperty,
# )
# from owlapy.iri import IRI

# from owlapy.owl_individual import OWLNamedIndividual

# import time
# from typing import Tuple, Set
# import pandas as pd
# from owlapy import owl_expression_to_dl
# from itertools import chain
# from argparse import ArgumentParser
# import os
# from tqdm import tqdm
# import random
# import itertools
# import ast
# from owlready2 import get_ontology
# # Set pandas options to ensure full output
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.colheader_justify', 'left')
# pd.set_option('display.expand_frame_repr', False)



# def concept_retrieval(retriever_func, c) -> Tuple[Set[str], float]:
#     start_time = time.time()
#     return {i.str for i in retriever_func.individuals(c)}, time.time() - start_time

# # removed_triples = {(stefan,type,father), (markus, has_child, anna)}

# path_true = "KGs/Family/father_enriched.owl"
# path_diminished = "KGs/Family/father_enriched_1.owl"

# onto = get_ontology(path_true).load()
# named_individuals = {ind.iri for ind in onto.individuals()}


# symbolic_kb_true = KnowledgeBase(path=path_true) # The True Knowledge base
# symbolic_kb_diminished = KnowledgeBase(path=path_diminished) # The Knowledge base without the omited triple


# neural_owl_reasoner = TripleStoreNeuralReasoner(path_neural_embedding="KGs_Family_father_enriched_1_owl_father",gamma=0.81)#(path_of_kb=path_diminished, gamma=0.081) This is to play with the removed (stefan,type,father)



# print(neural_owl_reasoner.model.predict(h="http://example.com/father#stefan", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))
# print(neural_owl_reasoner.model.predict(h="http://example.com/father#anna", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))
# print(neural_owl_reasoner.model.predict(h="http://example.com/father#heinz", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))
# print(neural_owl_reasoner.model.predict(h="http://example.com/father#michelle", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))
# print(neural_owl_reasoner.model.predict(h="http://example.com/father#martin", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))
# print(neural_owl_reasoner.model.predict(h="http://example.com/father#markus", r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type", t="http://example.com/father#father",logits=False))

# # expression =  OWLClass("http://example.com/father#father")

# # expression = OWLObjectSomeValuesFrom(
# #     property=OWLObjectProperty(IRI('http://example.com/father#', 'hasChild')),
# #     filler=OWLObjectOneOf([
# #         OWLNamedIndividual(IRI('http://example.com/father#', 'anna'))
# #     ])
# # )

# # expression = OWLObjectSomeValuesFrom(
# #     property=OWLObjectProperty(IRI('http://example.com/father#', 'hasChild')),
# #     filler=OWLClass(IRI('http://example.com/father#','person'))
# # )


# # A = OWLObjectSomeValuesFrom(
# #     property=OWLObjectProperty(IRI('http://example.com/father#', 'hasChild')),
# #     filler=OWLObjectOneOf([
# #         OWLNamedIndividual(IRI('http://example.com/father#', 'anna'))
# #     ])
# # )

# # B =  OWLClass("http://example.com/father#father")

# # expression = OWLObjectIntersectionOf([A,B])

# expression = OWLObjectUnionOf([OWLClass("http://example.com/father#father"), OWLClass("http://example.com/father#male")])



# print(owl_expression_to_dl(expression))

# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#stefan", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))
# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#anna", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))
# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#heinz", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))
# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#michelle", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))
# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#martin", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))
# # print(neural_owl_reasoner.model.predict(h="http://example.com/father#markus", r="http://example.com/father#hasChild", t="http://example.com/father#anna",logits=False))


# # exit(0)

# retrieval_y, runtime_y = concept_retrieval(symbolic_kb_true, expression) #The groundtruth retrieval
# # () Retrieve a set of inferred individuals and elapsed runtime.
# retrieval_ebr, runtime_ebr = concept_retrieval(neural_owl_reasoner, expression) #Retrieval for EBR 


# retrieval_ebr = {iri for iri in retrieval_ebr if iri in named_individuals}


# retrieval_fic, runtime_fic = concept_retrieval(symbolic_kb_diminished, expression) #Retrieval for EBR 


# # () Compute the Jaccard similarity.
# jaccard_sim_EBR = jaccard_similarity(retrieval_y, retrieval_ebr)
# jaccard_sim_fic = jaccard_similarity(retrieval_y, retrieval_fic)


# print(f"Jaccard similarity EBR: {jaccard_sim_EBR}")
# print(f"Jaccard similarity FIC: {jaccard_sim_fic}")

# print(f"retrieved EBR: {retrieval_ebr}")
# print(f"retrieved FIC: {retrieval_fic}")



