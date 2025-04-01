from scipy.special import rel_entr
from scipy.stats import entropy
import seaborn as sns
import pandas as pd
import sys
import os
from torch import nn
import sklearn
import pathlib
import pickle
import numpy as np
import torch.nn.functional as F
from scipy.special import rel_entr
from textblob import TextBlob    
import cv2
import re


def get_answer_with_probability(scores, gen_ids, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    filtered_tokens = [(x, y) for x, y in zip(gen_ids, scores) if x != pad_token_id]        # Remove pad tokens
    gen_ids = [x[0] for x in filtered_tokens]
    decoded_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in gen_ids]
    # print(gen_ids)
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)    
    answer_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    answer_token_logprobs = [x[1] for x in filtered_tokens]
    # print(answer_token_logprobs)
    if len(answer_tokens) == 1:      # SEP token only
        answer_tokens = ["", "[SEP]"]
        answer_token_logprobs = [-np.inf, 0.0]

    answer_token_logprobs_ = np.log(answer_token_logprobs).tolist()
    first_token_prob = answer_token_logprobs[0]
    # first_token_prob = np.exp(first_token_logprob)
    min_token_prob =  min(answer_token_logprobs)
    # min_token_prob = np.exp(min_token_logprob)
    mean_token_prob = sum(answer_token_logprobs[:-1]) / (len(answer_token_logprobs) - 1)
    # exp_mean_token_logprob = np.exp(mean_token_logprob)
    # mean_token_prob = sum(np.exp(answer_token_logprobs[:-1])) / (len(answer_token_logprobs) - 1)
    token_probs = answer_token_logprobs
    prod_token_probs = np.prod(answer_token_logprobs)
    norm_prod_token_probs = np.prod(answer_token_logprobs)**(1/len(answer_token_logprobs))
    logprobs_dict = {
        # "first_token_logprob": float(first_token_logprob),
        "first_token_prob": float(first_token_prob),
        # "min_token_logprob": float(min_token_logprob),
        "min_token_prob": float(min_token_prob),
        # "mean_token_logprob": float(mean_token_logprob),
        "mean_token_prob": float(mean_token_prob),
        # "exp_mean_token_logprob": float(exp_mean_token_logprob), 
        "answer_token_logprobs": answer_token_logprobs_,
        "token_probs": token_probs,
        "answer_tokens": answer_tokens,
        "prod_token_probs": float(prod_token_probs),
        "norm_prod_token_probs": float(norm_prod_token_probs),
    }
    return answer, decoded_tokens, logprobs_dict


def calculate_distance(p_logits, q_logits, p_prob, q_prob, logprobs_dict):
    # Entropy
    entropy_logit = entropy(p_logits).tolist()
    entropy_prob = entropy(p_prob).tolist()

    # KL distance
    KL_logit = np.sum(rel_entr(p_logits, q_logits)).tolist()
    KL_prob = np.sum(rel_entr(p_prob, q_prob)).tolist()

    #l2 distance
    l2_logits = np.linalg.norm(p_logits - q_logits)
    l2_prob = np.linalg.norm(p_prob - q_prob)

    #l1 distance
    l1_logits = np.linalg.norm(p_logits - q_logits, ord = 1)
    l1_prob = np.linalg.norm(p_prob - q_prob, ord = 1)

    # yes logit
    yeslogit_distance = abs(p_logits[0] - q_logits[0])
    yesprob_distance =  abs(p_prob[0] - q_prob[0])

    logprobs_dict = {
        "entropy_logit": float(entropy_logit),
        "entropy_prob": float(entropy_prob),
        "KL_logit": float(KL_logit),
        "KL_prob": float(KL_prob),
        "l2_logit": float(l2_logits),
        "l2_prob": float(l2_prob),
        "l1_logit": float(l1_logits), 
        "l1_prob": float(l1_prob),
        "yeslogit_distance": float(yeslogit_distance),
        "yesprob_distance": float(yesprob_distance),
    }
    return logprobs_dict


def get_answer_with_probability_beams(scores, gen_ids, tokenizer):
    pad_token_id = tokenizer.pad_token_id
    filtered_tokens = [(x, y) for x, y in zip(gen_ids, scores) if x != pad_token_id]        # Remove pad tokens
    gen_ids = [x[0] for x in filtered_tokens]
    decoded_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in gen_ids]
    # print(gen_ids)
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True)    
    answer_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    answer_token_logprobs = [x[1] for x in filtered_tokens]
    # print(answer_token_logprobs)
    if len(answer_tokens) == 1:      # SEP token only
        answer_tokens = ["", "[SEP]"]
        answer_token_logprobs = [-np.inf, 0.0]

    # answer_token_logprobs = np.log(answer_token_logprobs).tolist()
    first_token_logprob = answer_token_logprobs[0]
    first_token_prob = np.exp(first_token_logprob)
    min_token_logprob =  min(answer_token_logprobs)
    min_token_prob = np.exp(min_token_logprob)
    mean_token_logprob = sum(answer_token_logprobs[:-1]) / (len(answer_token_logprobs) - 1)
    exp_mean_token_logprob = np.exp(mean_token_logprob)
    mean_token_prob = sum(np.exp(answer_token_logprobs[:-1])) / (len(answer_token_logprobs) - 1)
    token_probs = np.exp(answer_token_logprobs).tolist()
    token_probs_log = answer_token_logprobs
    prod_token_probs = np.prod(np.exp(answer_token_logprobs))
    logprobs_dict = {
        "first_token_logprob": float(first_token_logprob),
        "first_token_prob": float(first_token_prob),
        "min_token_logprob": float(min_token_logprob),
        "min_token_prob": float(min_token_prob),
        "mean_token_logprob": float(mean_token_logprob),
        "mean_token_prob": float(mean_token_prob),
        "exp_mean_token_logprob": float(exp_mean_token_logprob), 
        "answer_token_logprobs": answer_token_logprobs,
        "token_probs": token_probs,
        "token_probs_log": token_probs_log,
        "answer_tokens": answer_tokens,
        "prod_token_probs": float(prod_token_probs)
    }
    return answer, decoded_tokens, logprobs_dict