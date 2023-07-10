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

from typing import Optional, List
import time
import random
import numpy as np
import openai

# set key before use
from transformers import T5Tokenizer, T5ForConditionalGeneration
from deeponto.align.bertmap import BERTMapPipeline


class LMPredictor:
    def __init__(self, choice: str, openai_key: Optional[str] = None, bertmap_model: Optional[BERTMapPipeline] = None):
        # choices: gpt, flan-t5, bertmap
        assert choice in ["gpt", "flan-t5", "bertmap"]
        self.model_type = choice
        if choice == "gpt":
            assert openai_key
            openai.api_key = openai_key
            self.predict = self.gpt_predict
        elif choice == "flan-t5":
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")
            self.predict = self.flan_t5_predict
        elif choice == "bertmap":
            # need to fine-tune first
            self.bertmap = bertmap_model

    def gpt_predict(self, input_text: str, max_retries: int = 5):
        """GPT prediction function with exponential waiting time."""
        retries = 0
        while True:
            try:
                completion = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=input_text,
                    max_tokens=3,
                    temperature=0,
                    logprobs=1,
                )
                answer = completion["choices"][0].text.strip()
                logprobs = completion["choices"][0].logprobs
                score = 0.0
                for tk, tk_score in zip(logprobs.tokens, logprobs.token_logprobs):
                    if "Yes" == tk:
                        score = np.exp(tk_score)
                        break
                    elif "No" == tk:
                        score = -np.exp(tk_score)  # assigning negative score
                        break
                break
            except:
                if retries == max_retries:
                    raise Exception("Retries exceeded")
                backoff_in_seconds = 1
                sleep = backoff_in_seconds * 2**retries + random.uniform(0, 1)
                time.sleep(sleep)
                retries += 1

        return answer, score

    def flan_t5_predict(self, input_text: str):
        """Flan-t5 prediction function (prediction scores are available)."""

        assert self.tokenizer
        assert self.t5

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.t5.generate(input_ids, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
        transition_scores = self.t5.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        input_length = 1 if self.t5.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        has_answer = False
        answer = None
        score = None
        for tok, tok_score in zip(generated_tokens[0], transition_scores[0]):
            # | token | token string | logits | probability
            tok_score = np.exp(tok_score.cpu().numpy())
            tok = self.tokenizer.decode(tok)
            # probs.append((tok, score))
            if "Yes" in tok or "yes" in tok:
                answer = "Yes"
                score = tok_score
                has_answer = True
                break
            if "No" in tok or "no" in tok:
                answer = "No"
                score = -tok_score  # assigning negative score
                has_answer = True
                break
        if not has_answer:
            answer = "No"
            score = 1.0

        return answer, score

    def bertmap_predict(self, src_class_labels: List[str], tgt_class_labels: List[str]):

        bertmap_score = self.bertmap.mapping_predictor.bert_mapping_score(src_class_labels, tgt_class_labels)
        # the bertmaplt score
        bertmaplt_score = self.bertmap.mapping_predictor.edit_similarity_mapping_score(
            src_class_labels, tgt_class_labels
        )

        return bertmap_score, bertmaplt_score
