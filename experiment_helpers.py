from ontolearn.owl_neural_reasoner import TripleStoreNeuralReasoner

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
import re
import ast
from owlready2 import get_ontology
import torch
from dicee.static_funcs import load_model
import numpy as np

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.class_expression import OWLClass
from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import RDFS, OWL

# Set pandas options to ensure full output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)


# Step 1: Load CSV files
def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def get_entity_df(base_path, label_na=-1, entity_filter=False):
    # File paths
    # base_path = f"./KGs_Family_family-benchmark_rich_background_owl_{pqr}"
    entities_file = f"{base_path}/DeCaL_entity_embeddings.csv"
    entity_map_file = f"{base_path}/entity_to_idx.p"

    # Load entity embeddings and map
    entities_df = load_csv(entities_file)  # First column is URI
    return entities_df


def get_class_expressions(dataset):
    input_owl_path = f"./KGs/{dataset}/{dataset.lower()}.owl"
    g = Graph()
    g.parse(input_owl_path, format='xml')

    FAMILY = Namespace(f"http://www.benchmark.org/{dataset.lower()}#")
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
        unique_classes.append(str(cls))

    return unique_classes


def get_individuals(dataset):
    input_owl_path = f"KGs/{dataset}/{dataset.lower()}.owl"

    g = Graph()
    g.parse(input_owl_path, format='xml')

    # Collect classes so we can ignore them when listing individuals
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))

    # Collect properties (optional but usually helpful)
    properties = set(g.subjects(RDF.type, RDF.Property)) | set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(
        g.subjects(RDF.type, OWL.DatatypeProperty))

    individuals = set()

    # Any subject with a type that isn't a class/property is considered an individual
    for s, p, o in g.triples((None, RDF.type, None)):
        if o not in classes and s not in classes and s not in properties and "#" in str(s):
            individuals.add(s)

    individual_uris = []
    for i, ind in enumerate(sorted(individuals)):
        individual_uris.append(str(ind))

    return individual_uris


def process_embeddings_df(dataset, df):
    individual_uris = get_individuals(dataset)
    class_uris = get_class_expressions(dataset)

    # Filter DataFrame to include only individuals and drop the URI column for embeddings
    ind_embeddings_df = df[df.iloc[:, 0].isin(individual_uris)].reset_index(drop=True)
    ind_embeddings = ind_embeddings_df.iloc[:, 1:]

    return individual_uris, class_uris, ind_embeddings.to_numpy(), ind_embeddings_df.iloc[:, 0].to_list()


def process_family_df(df):
    # Regex patterns
    # Corrected regex patterns
    individual_pattern = r'^http://www\.benchmark\.org/family#F\d+.*$'  # Starts with F + digits
    class_pattern = r'^http://www\.benchmark\.org/family#(?!F\d).*'  # Does NOT start with F + digits

    # Lists to store URIs
    individual_uris = []
    class_uris = []

    # Iterate over URIs in the first column
    for uri in df.iloc[:, 0]:
        if re.match(individual_pattern, uri):
            individual_uris.append(uri)
        elif re.match(class_pattern, uri):
            class_uris.append(uri)
        # else: ignore other URIs

    # Filter DataFrame to include only individuals and drop the URI column for embeddings
    ind_embeddings_df = df[df.iloc[:, 0].isin(individual_uris)].reset_index(drop=True)
    ind_embeddings = ind_embeddings_df.iloc[:, 1:]

    return individual_uris, class_uris, ind_embeddings.to_numpy(), ind_embeddings_df.iloc[:, 0].to_list()


def get_score(clifford_embedding, individuals, classes):
    path = f"/home/iroberts/projects/UPB_UBI/KGs_Family_family-benchmark_rich_background_owl_{clifford_embedding}"

    neural_owl_reasoner = TripleStoreNeuralReasoner(path_neural_embedding=path, gamma=0.1)
    score = neural_owl_reasoner.model.predict(h=individuals,
                                              r="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                                              t=classes, logits=False)

    return score


def get_relation_df(base_path, pqr):
    # File paths
    # base_path = f"./KGs_Family_family-benchmark_rich_background_owl_{pqr}"
    relations_file = f"{base_path}/DeCaL_relation_embeddings.csv"

    # Load entity embeddings and map
    relation_df = load_csv(relations_file)  # First column is URI
    return relation_df


def get_relation_type_embedding(df):
    # Regex patterns
    # Corrected regex patterns
    individual_pattern = r'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'  # Starts with F + digits

    # Lists to store URIs
    individual_uris = []
    class_uris = []

    # Iterate over URIs in the first column
    for uri in df.iloc[:, 0]:
        if individual_pattern == uri:
            # print(uri)
            individual_uris.append(uri)
        # else: ignore other URIs

    # Filter DataFrame to include only individuals and drop the URI column for embeddings
    type_embedding_df = df[df.iloc[:, 0].isin(individual_uris)].reset_index(drop=True)
    type_embedding = type_embedding_df.iloc[:, 1:]
    # print(type_embedding_df.iloc[:, 0])

    return type_embedding.to_numpy()

def get_class_embeddings(df, class_uris):
    # Filter DataFrame to include only individuals and drop the URI column for embeddings
    class_embeddings_df = df[df.iloc[:, 0].isin(class_uris)].reset_index(drop=True)
    class_embeddings = class_embeddings_df.iloc[:, 1:]
    classes = class_embeddings_df.iloc[:, 0].to_list()
    return class_embeddings.to_numpy(), classes



def construct_cl_multivector(x: torch.FloatTensor, re: int, p: int, q: int, r: int, model) -> tuple[
    torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Construct a batch of multivectors Cl_{p,q,r}(\mathbb{R}^d)
    Parameter
    ---------
    x: torch.FloatTensor with (n,d) shape
    Returns
    -------
    a0: torch.FloatTensor
    ap: torch.FloatTensor
    aq: torch.FloatTensor
    ar: torch.FloatTensor
    """
    # x = x.unsqueeze(dim=1).T
    batch_size, d = x.shape

    # (1) A_{n \times k}: take the first k columns
    a0 = x[:, :re].view(batch_size, re)
    # (2) B_{n \times p}, C_{n \times q}: take the self.k * self.p columns after the k. column
    if p > 0:
        ap = x[:, re: re + (re * p)].view(batch_size, re, p)
    else:
        ap = torch.zeros((batch_size, re, p), device=model.device)
    if q > 0:
        # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
        aq = x[:, re + (re * p):re + (re * p) + (re * q):].view(batch_size, re, q)
    else:
        aq = torch.zeros((batch_size, re, q), device=model.device)
    if r > 0:
        # (3) B_{n \times p}, C_{n \times q}: take the last self.r * self.q .
        ar = x[:, -(re * r):].view(batch_size, re, r)
    else:
        ar = torch.zeros((batch_size, re, r), device=model.device)
    return a0, ap, aq, ar


# This function will take embeddings and map them to a score for a class
# now I need to find the right
def pred_wrapper(h, t, model=None,pqr="0_0_0",base_path=None):
    r = get_relation_type_embedding(get_relation_df(base_path,pqr))
    r = r.reshape((r.shape[1],))
    # print(r)
    # Repeat r and t for each head entity
    n = h.shape[0]  # number of head entities

    head_ent_emb = torch.FloatTensor(h)  # shape: (n, d)
    rel_ent_emb = torch.FloatTensor(np.tile(r, (n, 1)))  # shape: (n, d)
    tail_ent_emb = torch.FloatTensor(np.tile(t, (n, 1)))  # shape: (n, d)

    # Construct multivectors for each batch element
    h0, hp, hq, hk = construct_cl_multivector(head_ent_emb, re=model.re, p=model.p, q=model.q, r=model.r, model=model)
    r0, rp, rq, rk = construct_cl_multivector(rel_ent_emb, re=model.re, p=model.p, q=model.q, r=model.r, model=model)
    t0, tp, tq, tk = construct_cl_multivector(tail_ent_emb, re=model.re, p=model.p, q=model.q, r=model.r, model=model)

    # (4) Compute a triple score based on interactions described by the basis 1.
    h0r0t0 = torch.einsum('br, br -> b', h0 * r0, t0)

    # (5) Compute a triple score based on interactions described by the bases of p {e_1, ..., e_p}.
    if model.p > 0:
        # Second term in Eq.16
        hp_rp_t0 = torch.einsum('brp, br  -> b', hp * rp, t0)
        # Eq. 17
        # b=e
        h0_rp_tp = torch.einsum('brp, erp -> b', torch.einsum('br,  brp -> brp', h0, rp), tp)
        hp_r0_tp = torch.einsum('brp, erp -> b', torch.einsum('brp, br  -> brp', hp, r0), tp)

        score_p = hp_rp_t0 + h0_rp_tp + hp_r0_tp
    else:
        score_p = 0

    # (5) Compute a triple score based on interactions described by the bases of q {e_{p+1}, ..., e_{p+q}}. Eq. 22
    if model.q > 0:
        # Third item in Eq 16.
        hq_rq_t0 = torch.einsum('brq, br  -> b', hq * rq, t0)
        # Eq. 18.
        h0_rq_tq = torch.einsum('br, brq  -> b', h0, rq * tq)
        r0_hq_tq = torch.einsum('br, brq  -> b', r0, hq * tq)
        score_q = - hq_rq_t0 + (h0_rq_tq + r0_hq_tq)
    else:
        score_q = 0

    if model.r > 0:
        # Eq. 18.
        h0_rk_tk = torch.einsum('br, brk  -> b', h0, rk * tk)
        r0_hk_tk = torch.einsum('br, brk  -> b', r0, hk * tk)
        score_r = (h0_rk_tk + r0_hk_tk)
    else:
        score_r = 0

    if model.p >= 2:
        sigma_pp = torch.sum(model.compute_sigma_pp(hp, rp), dim=[1, 2]).squeeze(-1)
    else:
        sigma_pp = 0
    if model.q >= 2:
        sigma_qq = torch.sum(model.compute_sigma_qq(hq, rq), dim=[1, 2]).squeeze(-1)
    else:
        sigma_qq = 0

    if model.r >= 2:
        sigma_rr = torch.sum(model.compute_sigma_rr(hk, rk), dim=[1, 2]).squeeze(-1)
    else:
        sigma_rr = 0

    if model.p >= 2 and model.q >= 2:
        sigma_pq = torch.sum(model.compute_sigma_pq(hp=hp, hq=hq, rp=rp, rq=rq), dim=[1, 2, 3]).squeeze(-1)
    else:
        sigma_pq = 0

    if model.p >= 2 and model.r >= 2:
        sigma_pr = torch.sum(model.compute_sigma_pr(hp=hp, hk=hk, rp=rp, rk=rk), dim=[1, 2, 3]).squeeze(-1)
    else:
        sigma_pr = 0
    if model.q >= 2 and model.r >= 2:
        sigma_qr = torch.sum(model.compute_sigma_qr(hq=hq, hk=hk, rq=rq, rk=rk), dim=[1, 2, 3]).squeeze(-1)
    else:
        sigma_qr = 0

    dot = h0r0t0 + score_p + score_q + score_r + sigma_pp + sigma_qq + sigma_rr + sigma_pq + sigma_qr + sigma_pr
    prob = torch.sigmoid(dot).numpy()
    # prob = threshold_sigmoid(dot.numpy())

    return np.stack([1 - prob, prob], axis=1)

# def threshold_sigmoid(x, gamma=, k=1):
#     return 1 / (1 + np.exp(-k * (x - gamma)))
