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
            rel = "="
            # penalise the score for answering "No"
            if answer == "No":
                score -= 1.0
                rel = "!="
            mapping = EntityMapping(src_ref, tgt_cand, rel, score)
            cur_mappings.append(mapping)
            # final prediction determines by "Yes" and/or threshold
            if answer == "Yes" and score >= threshold:
                final_preds.append(mapping)
        ranked_preds[src_ref, tgt_ref] = EntityMapping.sort_entity_mappings_by_score(cur_mappings)

    return final_preds, ranked_preds

def evaluate(final_preds, ranked_preds, refs, include_latex: bool = False):

    yes_hit1 = 0
    no_hit1 = 0

    _ranked = []
    for (src_ref, tgt_ref), cand_mappings in ranked_preds.items():
        if cand_mappings[0].tail == tgt_ref:
            yes_hit1 += 1
        elif cand_mappings[0].relation == "!=" and tgt_ref == "UnMatched":
            no_hit1 += 1
        else:
            pass
            # print((src_ref, tgt_ref))
        _ranked.append((ReferenceMapping(src_ref, tgt_ref, "="), cand_mappings))

    matching_scores = AlignmentEvaluator().f1(final_preds, refs)

    mrr = AlignmentEvaluator().mean_reciprocal_rank(_ranked[:50])

    all_scores = matching_scores
    all_scores["Hits@1+"] = yes_hit1 / 50
    all_scores["Hits@1-"] = no_hit1 / 50
    all_scores["MRR"] = mrr

    if include_latex:
        print(" & ".join([str(round(s, 3)) for s in all_scores.values()]))

    return all_scores
