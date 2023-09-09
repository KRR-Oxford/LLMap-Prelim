#    Copyright 2023 Yuan He

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#   limitations under the License.

from typing import List
from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline


def load_ontos(src_onto_file: str, tgt_onto_file: str):

    # load ontologies
    src_onto = Ontology(src_onto_file)
    tgt_onto = Ontology(tgt_onto_file)

    # load default BERTMap config for the annotation properties
    config = BERTMapPipeline.load_bertmap_config()

    # build annotation index {class_iri: class_labels}
    src_annotation_index, _ = src_onto.build_annotation_index(config.annotation_property_iris)
    tgt_annotation_index, _ = tgt_onto.build_annotation_index(config.annotation_property_iris)

    return src_onto, tgt_onto, src_annotation_index, tgt_annotation_index, config


# controlling input labels because of the window size limit of Flan-T5
def truncate_labels(labels, cut_off: int = 3):
    """To prevent token overflow, truncate the labels to the set of longest ones."""
    labels = list(labels)
    if len(labels) >= cut_off:
        labels.sort(key=len, reverse=True)
    return labels[:cut_off]


def get_parent_labels(ontology: Ontology, annotation_index: dict, class_iri: str):
    """Get the parent concepts of a given concept."""
    concept = ontology.get_owl_object_from_iri(class_iri)
    concept_parents = ontology.get_asserted_parents(concept, named_only=True)
    concept_parent_labels = []
    for p in concept_parents:
        # select just one label for each parent concept
        concept_parent_labels += truncate_labels(annotation_index[str(p.getIRI())], cut_off=1)
    concept_parent_labels = set(concept_parent_labels)
    return list(concept_parent_labels)


def get_child_labels(ontology: Ontology, annotation_index: dict, class_iri: str):
    """Get the child concepts of a given concept."""
    concept = ontology.get_owl_object_from_iri(class_iri)
    concept_children = ontology.get_asserted_children(concept, named_only=True)
    concept_children_labels = []
    for c in concept_children:
        # select just one label for each child concept
        concept_children_labels += truncate_labels(annotation_index[str(c.getIRI())], cut_off=1)
    concept_children_labels = set(concept_children_labels)
    return list(concept_children_labels)


def concept_template(title: str, list_of_names: List[str], compact_list: bool):
    
    # compact list better for flan-t5
    if compact_list:
        return f"{title}: {list_of_names}\n"

    # point-by-point list better for chatgpt
    result = f"{title}:\n"
    for i, n in enumerate(list_of_names):
        result += f"- {n}\n"
    return result

def integrated_template(
    src_concept_labels: list, 
    tgt_concept_labels: list, 
    src_parent_labels: list = None, 
    tgt_parent_labels: list = None, 
    src_child_labels: list = None, 
    tgt_child_labels: list = None,
    compact_list: bool = False,
):
    v = concept_template("Source Concept Names", src_concept_labels, compact_list)
    has_parent_child = False
    if src_parent_labels:
        v += concept_template("Parent Concepts of the Source Concept", src_parent_labels, compact_list)
        has_parent_child = True
    if src_child_labels:
        v += concept_template("Child Concepts of the Source Concept", src_child_labels, compact_list)
        has_parent_child = True
    v +="\n"
    v += concept_template("Target Concept Names", tgt_concept_labels, compact_list)
    if tgt_parent_labels:
        v += concept_template("Parent Concepts of the Target Concept", tgt_parent_labels, compact_list)
        has_parent_child = True
    if tgt_child_labels:
        v += concept_template("Child Concepts of the Target Concept", tgt_child_labels, compact_list)
        has_parent_child = True
    v +="\n"
    if not has_parent_child:
        v = "Given the lists of names associated with two concepts, your task is to determine whether these concepts are identical or not. Consider the following:\n\n" + v
        v += "Analyze the names provided for each concept and provide a conclusion on whether these two concepts are identical or different (\"Yes\" or \"No\") based on their associated names."
    else:
        v = "Given the lists of names and hierarchical relationships associated with two concepts, your task is to determine whether these concepts are identical or not. Please consider the following:\n\n" + v
        v += "Analyze the names and the hierarchical information provided for each concept, and provide a conclusion on whether these two concepts are identical or different (\"Yes\" or \"No\") based on their associated names and hierarchical relationships."
    return v
