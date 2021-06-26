# GuidedRec
This is our implementation for the SIGIR 2021 paper:

Rashed, Ahmed, Josif Grabocka, and Lars Schmidt-Thieme. "A Guided Learning Approach for Item Recommendation via Surrogate Loss Learning (SIGIR). 2021.
## Enviroment for GuidedRec, Surrogate Loss and  Logloss
	* pandas==1.0.3
	* tensorflow==1.14.0
	* matplotlib==3.1.3
	* numpy==1.18.1
	* six==1.14.0
	* scikit_learn==0.23.1
  
### Commands
#### GraphRec Model
##### with logloss
* python GraphRec.py 0 45
##### with GuidedRec
* python GraphRec.py 1 45
#### surrogate Only
* python GraphRecSurrogate.py 1 45

#### NueMF Model
##### with logloss
* python NueMF.py 0 45


## Enviroment for TFRanking Losses 
	* numpy==1.18.1
	* six==1.14.0
	* matplotlib==3.1.3
	* tensorflow==2.3.0
	* pandas==1.0.3
	* scikit_learn==0.23.1
	* tensorflow_addons==0.10.0
	* tensorflow_ranking==0.3.0

### Commands
#### GraphRec Model
##### with gumbel_approx_ndcg_loss
* python GraphRecTFRank.py  0 45 "\'gumbel_approx_ndcg_loss\'"

#####  with approx_ndcg_loss
* python GraphRecTFRank.py  0 45 "\'approx_ndcg_loss\'"

##### with list_mle_loss
* python GraphRecTFRank.py  0 45 "\'list_mle_loss\'"

##### with softmax_loss
* python GraphRecTFRank.py  0 45 "\'softmax_loss\'"

##### with pairwise_logistic_loss
* python GraphRecTFRank.py  0 45 "\'pairwise_logistic_loss\'"

##### with neural_sort_cross_entropy_loss
* python GraphRecTFRank.py  0 45 "\'neural_sort_cross_entropy_loss\'"

## Note: All scripts require a gpu. Please change the device <DEVICE = "/gpu:0"> to cpu if you have no gpu
