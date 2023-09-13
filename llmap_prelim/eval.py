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

from deeponto.align.mapping import ReferenceMapping, EntityMapping
from deeponto.align.evaluation import AlignmentEvaluator


def unpack_results_for_llm(result_dict: dict, threshold: float = 0.0):

    final_preds = []  # set of final predictions
    ranked_preds = dict()  # all yes/no predictions ranked by their scores

    for (src_ref, tgt_ref), v in result_dict.items():
        cur_mappings = []
        for tgt_cand, (answer, score) in v.items():
            mapping = EntityMapping(src_ref, tgt_cand, "=", score)
            # final prediction determines by "Yes" and/or threshold
            if ("Yes" in answer or "yes" in answer or "are identical" in answer) and score >= threshold:
                final_preds.append(mapping)
            else:
                mapping.relation = "!="   
            cur_mappings.append(mapping)
                
        ranked_preds[src_ref, tgt_ref] = EntityMapping.sort_entity_mappings_by_score(cur_mappings)

    return final_preds, ranked_preds

def unpack_results_for_bertmap(result_dict: dict, bertmap_threshold: float = 0.0, bertmaplt_threshold: float = 0.0):
    
    bertmap_final_preds = []
    bertmap_ranked_preds = dict()
    
    bertmaplt_final_preds = []
    bertmaplt_ranked_preds = dict()
    
    for (src_ref, tgt_ref), v in result_dict.items():
        
        cur_bertmap_mappings = []
        cur_bertmaplt_mappings = []
        
        for tgt_cand, (bertmap_score, bertmaplt_score) in v.items():
            
            bertmap_mapping = EntityMapping(src_ref, tgt_cand, "=", bertmap_score)
            cur_bertmap_mappings.append(bertmap_mapping)
            if bertmap_score >= bertmap_threshold:
                bertmap_final_preds.append(bertmap_mapping)
            else:
                bertmap_mapping.relation = "!="
            
            bertmaplt_mapping = EntityMapping(src_ref, tgt_cand, "=", bertmaplt_score)
            cur_bertmaplt_mappings.append(bertmaplt_mapping)
            if bertmaplt_score >= bertmaplt_threshold:
                bertmaplt_final_preds.append(bertmaplt_mapping)
            else:
                bertmaplt_mapping.relation = "!="
                    
        bertmap_ranked_preds[src_ref, tgt_ref] = EntityMapping.sort_entity_mappings_by_score(cur_bertmap_mappings)
        bertmaplt_ranked_preds[src_ref, tgt_ref] = EntityMapping.sort_entity_mappings_by_score(cur_bertmaplt_mappings)

    return bertmap_final_preds, bertmap_ranked_preds, bertmaplt_final_preds, bertmaplt_ranked_preds
    

def evaluate(final_preds, ranked_preds, refs, include_latex: bool = False):

    hits1 = 0
    reject = 0

    _ranked = []
    for (src_ref, tgt_ref), cand_mappings in ranked_preds.items():
        if cand_mappings[0].tail == tgt_ref:
            hits1 += 1
        elif cand_mappings[0].relation == "!=" and tgt_ref == "UnMatched":
            reject += 1
        else:
            pass
            # print((src_ref, tgt_ref))
        _ranked.append((ReferenceMapping(src_ref, tgt_ref, "="), cand_mappings))

    matching_scores = AlignmentEvaluator().f1(final_preds, refs)

    mrr = AlignmentEvaluator().mean_reciprocal_rank(_ranked[:50])

    all_scores = matching_scores
    all_scores["Hits@1"] = hits1 / 50
    all_scores["MRR"] = mrr
    all_scores["RR"] = reject / 50

    if include_latex:
        print(" & ".join([str(round(s, 3)) for s in all_scores.values()]))

    return all_scores
