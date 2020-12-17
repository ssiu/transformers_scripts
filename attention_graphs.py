#!/usr/bin/python3
"""
Generate attention graphs from a sentence
"""

from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Encoder:
    def __init__(self, model, max_len=256):
        """
        Params:
            model:
            num_layers: integer <= 13
            max_len
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model, normalization=True)
        self.config = AutoConfig.from_pretrained(model, output_attentions=True, output_hidden_states=False)
        self.model = AutoModel.from_pretrained(model, config=self.config)
        self.max_len = max_len
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, text):
        """
        Params:
            text: a sentence string
        Returns:
            all_hidden_states: numpy array of size [seq_len, num_layers, hidden_size=768]
        """
        with torch.no_grad():
            inputs = self.tokenizer.encode(text)
            #print('Tokens are ', self.tokenizer.convert_ids_to_tokens(inputs))
            input_ids = torch.tensor(inputs).unsqueeze(0).to(self.device)
            outputs = self.model(input_ids)
            """
            output[0]: torch.size([batch_size, seq_len, hidden_size]) output of the last layer
            output[1]: torch.size([batch_size, hidden_size]) CLS
            output[2]: array of size 13 
                       each element is torch.size([batch_size, num_heads, seq_len, seq_len])
                       output[2][0], output[2][12] are the output of the first and last layer
            """
            a = np.array([output.cpu().detach().numpy() for output in outputs[2][::-1]])

            num_layers, batch_size, num_heads, seq_len, seq_len = a.shape

            return np.reshape(a, (num_layers, num_heads, seq_len, seq_len)), [num_layers, num_heads, seq_len, seq_len]


def plot_attention(sentence, model, path):
    '''
    Params:
        sentence: input for the transformer
        model: the pretrained model
        path: file name to save the attention plots
    '''
    print(f"Inputting sentence into {model}...")
    model = Encoder(model)
    outputs, sizes = model.encode(sentence)
    print(f"Plotting attention values...")
    indices = pd.MultiIndex.from_product((range(sizes[0]), range(sizes[1]), range(sizes[2]), range(sizes[3])),
                                         names=('num_layers', 'num_heads', 'queries', 'keys'))

    df = pd.DataFrame(outputs.ravel(), index=indices, columns=('value',)).reset_index()

    fg = sns.FacetGrid(df, row='num_layers', col='num_heads', margin_titles=False,
                       legend_out=True)  # , sharex=True, sharey=True)

    fg.map_dataframe(draw_heatmap, 'keys', 'queries', 'value', cbar=False, square=True)
    fg.set_axis_labels("", "")
    fg.add_legend()
    for ax in fg.axes.flatten():
        ax.set_title('')
    plt.savefig(path)
    print(f"plots saved to {path}.")
    return 0


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, xticklabels=False, yticklabels=False, cmap='YlOrRd', **kwargs)
    #'viridis'
    #'YlOrRd'
    #Reds
    #ax.set_xlabel('')
    #ax.set_ylabel('')


#'bert-base-uncased'
#'allenai/scibert_scivocab_uncased'
#'dmis-lab/biobert-v1.1'
#'vinai/bertweet-base'

bert_sentence = 'Australia\'s capital is Canberra, and its largest city is Sydney.'
scibert_sentence = 'Experimental infection of a US spike-insertion deletion porcine epidemic diarrhea virus in conventional nursing piglets and cross-protection to the original US PEDV infection.'
biobert_sentence = 'The primary objective of the study is to compare the overall response rate (inclusive of complete response, partial response and hematologic improvement) per IWG 2006 criteria in patients with higher risk MDS treated with azacitidine with or without deferasirox achieved over the course of one year. Hematologic improvement must be maintained for at least 8 weeks.'
bertweet_sentence = 'Still time 2 enter my @ModereUS sample #giveaway on my @YouTube channel! Check it out here http://t.co/4BUMuM2Rqh #modere #bbloggers #beauty'

plot_attention(bert_sentence, 'bert-base-uncased', 'bert.png')
plot_attention(scibert_sentence, 'allenai/scibert_scivocab_uncased', 'scibert.png')
plot_attention(biobert_sentence, 'dmis-lab/biobert-v1.1', 'biobert.png')
plot_attention(bertweet_sentence, 'vinai/bertweet-base', 'bertweet.png')



