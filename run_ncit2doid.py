import os
import sys

main_dir = os.getcwd().split("LLMap")[0] + "LLMap"
sys.path.append(main_dir)

from deeponto.align.bertmap import BERTMapPipeline
from deeponto.utils import read_table
from llmap_prelim.processing import load_ontos
from llmap_prelim.models import LMPredictor
from llmap_prelim import run_experiments
import click

@click.command()
@click.option("-m", "--model_type", type=str)
@click.option("-k", "--api_key", type=str, default=None)
@click.option("-s", "--with_structural_context", type=bool, default=False)
def run(model_type, api_key, with_structural_context):

    src_onto_file = f"{main_dir}/data/ncit2doid/ncit.owl"
    tgt_onto_file = f"{main_dir}/data/ncit2doid/doid.owl"
    test_cand_file = f"{main_dir}/data/ncit2doid/test_cands.tsv"
    result_file = f"./{model_type}_ncit2doid_results.pkl"
    if with_structural_context:
        result_file = f"./{model_type}_ncit2doid_results_struct.pkl"
    
    src_onto, tgt_onto, src_annotation_index, tgt_annotation_index, config = load_ontos(src_onto_file, tgt_onto_file)
    test_cands = read_table(test_cand_file)
    
    bertmap = None
    if model_type == "bertmap":
        config.global_matching.enabled = False
        config.output_path = "ncit2doid.us/"
        bertmap = BERTMapPipeline(src_onto, tgt_onto, config)

    predictor = LMPredictor(model_type, api_key, bertmap)

    run_experiments(
        src_onto,
        tgt_onto,
        src_annotation_index,
        tgt_annotation_index,
        predictor,
        test_cands,
        result_file,
        with_structural_context=with_structural_context,
    )


if __name__ == "__main__":
    run()
