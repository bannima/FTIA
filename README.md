#FTIA
FTIA: Fusion of Task-specific Information based Attention Mechanism for Text Representation Learning

##1.Environmental Requirements
- python=2.7
- pytorch=1.0
- GPU optional but recommended !

Please download GloVe word vectors in dir src/wordvectors/.static_vector_cache. More details can be found
in https://nlp.stanford.edu/projects/glove/

##2.Datasets
Nine text classification datasets are used in our experiment.

|Dataset | Classes| Source
---------| -------------|---------
AGNews | 4|http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
20News|20 |http://qwone.com/~jason/20Newsgroups/
MR|2|http://www.cs.cornell.edu/people/pabo/movie-review-data/
CR|2 | https://www.cs.uic.edu/%18liub/FBS/sentiment-analysis.html
MPQA|2 |http://mpqa.cs.pitt.edu/
Subj|2|https://github.com/mhjabreel/CharCNN/tree/master/data
TREC| 6|http://cogcomp.cs.illinois.edu/Data/QA/QC/
SST-1| 5|https://nlp.stanford.edu/sentiment/
SST-2|2| https://nlp.stanford.edu/sentiment/


##3.FTIA model
The FTIA model are constructed as follows:

![FTIA](figs/FTIA_model.png "FTIA Model")


##4.Results
###4.1 Accuracy

The experiment accuracy for each dataset and model are organized in the following table.


 |模型 | CR | MR | SST-1 | SST-2 | MPQA | TREC | Subj
 ------ | ------ | ------ |------ | ------ | ------ |------ | ------ | 
|TextCNN | 67.02 |93.21|31.67|67.22|87.35|79.8|86.27
|LSTM|77.98|90.92|43.3|81.99|92.68|88.8|92.0
|BiGRU|72.5|99.73|36.47|74.74|99.0|83.0|87.33
|LSTMAtt|73.83|99.79|37.51|75.62|98.72|81.6|87.47
|RCNN|75.95|100.0|38.91|74.46|99.06|87.4|86.97
|SelfAtt|74.71|100.0|37.01|74.92|99.0|85.8|86.53
|FTIA-rand|74.54|96.14|38.42|74.9|95.69|84.0|86.23
|FTIA-static|77.54|92.44|43.62|81.93|94.25|86.4|92.43
|FTIA-non-static|79.95|98.73|43.21|81.49|97.66|86.6|92.7


###4.2 Attention Weights

We randomly visualized the five movie reviews with corresponding learned attention weights in MR dataset, for understanding
the FTIA model. The figure is shown as follow:

![AttentionWeights](./figs/Attention_weights_visulaization_penalty01.png)

###4.3 Text Representations

For better understanding of learned text representations, we use the data reduction tool
t-SNE to transform orignal high dimensional text representations to 2-D vectors. Furthermore,
we use python scatter tool to visualize the FBIA 2-D vectors in different epochs, with the compared LSTMAtt text representations.

![TextRepresentations](./figs/Epoch1_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch5_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch10_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch20_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch30_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch40_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch70_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")

![TextRepresentations](./figs/Epoch100_FBIA_LSTM_tSNE_Comp.png "Text Representations Visualization")


Please note that all experiment results are stored in all_results.csv file. And
more experiment details can be found in our original paper.