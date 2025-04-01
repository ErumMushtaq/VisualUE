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

# Functions needed for eval

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(?<=\d)(\,)+(?=\d)")
puncStrip = re.compile(
    r"(?<=[ \\;/\"`\[\](){}<>@=+_\-,?!])([\\;/\"`\[\](){}<>@=+_\-,?!])|([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[ \\;/\"`\[\](){}<>@=+_\-,?!])"
)
puncStrip2 = re.compile(r"(?<=[a-zA-Z])([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[a-zA-Z])")
puncStripBegin = re.compile(r"\A([ \\;/\"`\[\](){}<>@=+_\-,?!]+)(?=[a-zA-Z0-9 ])")
puncStripEnd = re.compile(r"(?<=[a-zA-Z0-9 ])([ \\;/\"`\[\](){}<>@=+_\-,?!]+)\Z")
spaceCleanup = re.compile(r"([ ]+)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def processPunctuation(inText):
    outText = puncStripBegin.sub("", inText)
    outText = puncStripEnd.sub("", outText)
    outText = commaStrip.sub("", outText)
    outText = puncStrip.sub(" ", outText)
    outText = spaceCleanup.sub(" ", outText)
    outText = puncStrip2.sub(" ", outText)
    outText = puncStrip2.sub("", outText)
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText

def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText

def AUROC(confidence_scores, correctness, label):
    auroc_score = sklearn.metrics.roc_auc_score(1 - np.array(correctness),
                                                            1-np.array(confidence_scores))
    print("AUROC "+label+" "+str(sklearn.metrics.roc_auc_score(1 - np.array(correctness),
                                                            1-np.array(confidence_scores))))
    # wandb.log({"AUROC "+label: auroc_score, " temp number ": 1000})

def confidence_bin_plots(confidence_correct, confidence_incorrect, label_name):
    plt.figure()
    bin_count = 30
    scaling_ratio = 1
    # hist, bins = np.histogram(np.array(confidence_correct), range=(0, 1), bins=bin_count,density=True)

    hist, bins = np.histogram(np.array(confidence_correct), range=(np.min(np.array(confidence_correct)), np.max(np.array(confidence_correct))), bins=bin_count, density=True)

    scaling_ratio= 100/sum(hist) #normalize to sum to 100  
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist*scaling_ratio, align='center', width=width, zorder=1, color='blue', alpha=0.5,label='Correct')
    hist, bins = np.histogram(np.array(confidence_incorrect), range=(np.min(np.array(confidence_incorrect)), np.max(np.array(confidence_incorrect))), bins=bin_count, density=True)
    scaling_ratio= 100/sum(hist) #normalize to sum to 100 (due to bin count)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist*scaling_ratio, align='center', width=width, zorder=1, color='black', alpha=0.5,label='Incorrect')
    plt.ylim(0,100)

    plt.xlabel(label_name)
    plt.ylabel("Percent")
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig(label_name+'bin_plots.pdf')
    # plt.savefig("plots/"+label_name+'bin_plots.png')  
    # wandb.log({label_name+"bin_plots": wandb.Image(plt)})

def density_plots(confidence_correct_topk_kl_same, confidence_incorrect_topk_kl_same, confidence_correct_topk_kl_notsame, confidence_incorrect_topk_kl_notsame, label):
#confidence_correct_topk_kl_same, confidence_incorrect_topk_kl_same, confidence_correct_topk_kl_notsame, confidence_incorrect_topk_kl_notsame
    # np.random.seed(42)
    colors = ["#d22424", "#971d78", "#e38322", "#166963"]
    plt.figure(figsize=(10, 6))
    # Create histogram to get frequency counts
    counts, bin_edges = np.histogram(confidence_correct_topk_kl_same, bins=40)
    # Midpoints of bins for plotting
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
   

    data = pd.DataFrame({'score': bin_centers, 'frequency': counts})
    print(len(data))
    # Creating the density plot
    sns.histplot(confidence_correct_topk_kl_same, kde=True, stat="count", color=colors[0],
        # fill=True,
        binwidth=0.05, label='Correct, but Same')
    # sns.kdeplot(
    #     data= data,
    #     x=data['score'],
    #     # y=data['frequency'],
    #     weights=len(data)*len(data),
    #     color=colors[0],
    #     # fill=True,
    #     label='Correct, but Same'
    # )

    # counts, bin_edges = np.histogram(confidence_incorrect_topk_kl_same, bins=40)
    # # Midpoints of bins for plotting
    # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    sns.histplot(confidence_incorrect_topk_kl_same, kde=True, stat="count", color=colors[1],  
        # fill=True,
       binwidth=0.05,label='Incorrect but Same')

    # data = pd.DataFrame({'score': bin_centers, 'frequency': counts})
    # # Creating the density plot
    # sns.kdeplot(
    #     data= data,
    #     x=data['score'],
    #     # y=data['frequency'],
    #     weights=len(data)*len(data),
    #     color=colors[1],
    #     # fill=True,
    #     label= 'Incorrect but Same'
    # )

    # print(confidence_correct_topk_kl_notsame)
    sns.histplot(confidence_correct_topk_kl_notsame, kde=True, stat="count", color=colors[2],  
        # fill=True,
        binwidth=0.05,label='Correct but Different')
    # counts, bin_edges = np.histogram(confidence_correct_topk_kl_notsame, bins=40)
    # # Midpoints of bins for plotting
    # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # data = pd.DataFrame({'score': bin_centers, 'frequency': counts})
    # sns.histplot(data, kde=True, color=colors[2],stat="density")
    # Creating the density plot
    # sns.kdeplot(
    #     data= data,
    #     x=data['score'],
    #     # y=data['frequency'],
    #     weights=len(data)*len(data),
    #     color=colors[2],
    #     # fill=True,
    #     label= 'Correct but Different'
    # )

    print(bin_centers)
    print(confidence_incorrect_topk_kl_notsame)
    # counts, bin_edges = np.histogram(confidence_incorrect_topk_kl_notsame, bins=40)
    sns.histplot(confidence_incorrect_topk_kl_notsame, kde=True, stat="count", color=colors[3],  
        # fill=True,
        binwidth=0.05,label='Incorrect and Different')

    # Midpoints of bins for plotting
    # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # data = pd.DataFrame({'score': bin_centers, 'frequency': counts})
    # # Creating the density plot
    # sns.kdeplot(
    #     data= data,
    #     x=data['score'],
    #     # y=data['frequency'],
    #     color=colors[3],
    #     weights=len(data)*len(data),
    #     # fill=True,
    #     label='Incorrect and Different'
    # )

    print(bin_centers)
    # Labels and title
    plt.title('Frequency vs. Scoring Function')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.legend()
    # plt.ylim([0,1])
    # plt.colorbar(label='Density')
    plt.show()