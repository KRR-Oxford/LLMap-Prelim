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
from deeponto.onto import Ontology
from deeponto.utils import FileUtils
from collections import defaultdict
import enlighten
from pandas import DataFrame
from .models import LMPredictor
from .processing import truncate_labels, integrated_template, get_parent_labels, get_child_labels


def run_experiments(
    src_onto: Ontology,
    tgt_onto: Ontology,
    src_annotation_index: dict,
    tgt_annotation_index: dict,
    predictor: LMPredictor,
    test_cands: DataFrame,
    result_file: str,
    with_structural_context: bool = False,
):

    try:
        result_dict = FileUtils.load_file(result_file)
    except:
        result_dict = defaultdict(dict)

    enlighten_manager = enlighten.get_manager()
    progress_bar = enlighten_manager.counter(total=len(test_cands), desc="Mapping Prediction", unit="per src class")

    for _, dp in test_cands.iterrows():

        src_class_iri = dp["SrcEntity"]
        src_class_labels = truncate_labels(src_annotation_index[src_class_iri], 3)
        tgt_class_iri = dp["TgtEntity"]
        tgt_cands = eval(dp["TgtCandidates"])
        temp_progress_bar = enlighten_manager.counter(
            total=len(tgt_cands), desc="Mapping Prediction", unit="per tgt candidate"
        )

        src_class_parents = (
            get_parent_labels(src_onto, src_annotation_index, src_class_iri) if with_structural_context else None
        )
        src_class_children = (
            get_child_labels(src_onto, src_annotation_index, src_class_iri) if with_structural_context else None
        )

        for tgt_cand_iri in tgt_cands:
            # skip predicted candidates (this is especially useful for GPT-3.5 as the connection is not stable)
            if tgt_cand_iri in result_dict[src_class_iri, tgt_class_iri].keys():
                continue
            tgt_cand_labels = truncate_labels(tgt_annotation_index[tgt_cand_iri], 3)

            tgt_cand_parents = (
                get_parent_labels(tgt_onto, tgt_annotation_index, tgt_cand_iri) if with_structural_context else None
            )
            tgt_cand_children = (
                get_child_labels(tgt_onto, tgt_annotation_index, tgt_cand_iri) if with_structural_context else None
            )

            if predictor.model_type != "bertmap":
                # compact_list = predictor.model_type == "flan-t5"
                compact_list = False
                input_text = integrated_template(
                    src_class_labels,
                    tgt_cand_labels,
                    src_class_parents,
                    tgt_cand_parents,
                    src_class_children,
                    tgt_cand_children,
                    compact_list=compact_list,
                )
                answer, score = predictor.predict(input_text)
                result_dict[src_class_iri, tgt_class_iri][tgt_cand_iri] = (answer, score)
            else:
                # the bertmap model has no "answer" but it can produce two types of scores
                bertmap_score, bertmaplt_score = predictor.predict(src_annotation_index[src_class_iri], tgt_annotation_index[tgt_cand_iri])
                result_dict[src_class_iri, tgt_class_iri][tgt_cand_iri] = (bertmap_score, bertmaplt_score)

            temp_progress_bar.update()

        progress_bar.update()
        FileUtils.save_file(result_dict, result_file)
