
# Towards Monosemanticity

This repository reproduces results of [Anthropic's Monosemanticity paper](https://transformer-circuits.pub/2023/monosemantic-features/). It is still a work in progress, but I am making it public to invite feedback and comments. Please feel free to share your feedback. Thank you!

At the current stage, one can train a sparse autoencoder and analyze its features following the steps in the Reproduction section. A brief analysis of features has seemed promising. While I found most of the neurons to be dead (a few of the possible reasons for this are mentioned in the 'Possible Improvements' section below), I found ~1750 of 4096 neurons to be in the high density cluster.

<div align="center">
    <img src="assets/feature_density_histogram.png" width="300">
</div>

These neurons and their top 10 activation values along with tokens and contexts are given in [high_density_neurons.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/high_density_neurons.html). I inspected the first 20 features manually, out of which I could interpret 19. I have written the common theme shared by top 10 activations for these 19 neurons in the HTML file. I also discovered more than 650 single-token features with the help of a Python function. (You may CTRL+F/CMD+F "single-token" on the HTML page to see them.) Some of these may be token-in-context features, but I have not checked that for individual neurons. 

An interesting example is of the pair of neurons # 215 and # 2601. Both of these neurons fire on Cyrilic script, but one of them fires on vowels (е, у, и, а) and the other one fires on consonants (л, в, р, т). 

<p align="middle">
  <img src="./assets/neuron_215.png" width="230" /> 
  <img src="./assets/neuron_2601.png" width="250" /> 
</p>


The loss curves and feature density histograms for the best training run so far are available on this [Weights and Biases page](https://wandb.ai/shehper/sparse-autoencoder-openwebtext-public).

## reproducing results

**step 0: make a virtual environment and install required packages**

Clone the repository and change the directory.
```
https://github.com/shehper/monosemantic.git && cd monosemantic
```

Make a new virtual environment, and activate it.
```
python -m venv ./env
source ./env/bin/activate
```

Install packages from requirements.txt.
```
pip install -r requirements.txt
```

I used Python 3.9 for this project. If you have an older version of OpenSSL on your machine, you will notice that downloading and tokenizing dataset in Step 1 will return a compatibility error between the versions of urllib3 and OpenSSL. In this case, you may upgrade OpenSSL or downgrade sentry-sdk and urllib3 to older versions as follows.

```
pip install sentry-sdk==1.29.2 # try only if prepare.py in Step 1 returns ImportError for urllib3
pip install urllib3==1.26.15 # try only if prepare.py in Step 1 returns ImportError for urllib3
```

**step 1: train a one-layer transformer model**

I used [nanoGPT](https://github.com/karpathy/nanoGPT) to train a one-layer transformer. The required code is in the 'transformer' subfolder of this repository. 

In order to train this transformer model, first move to the 'transformer' subdirectory.
```
cd transformer 
```

Next, download and tokenize the OpenWebText dataset as follows. (If it gives any import errors, please look at the possible solution provided in Step 0.)

```
python data/openwebtext/prepare.py 
```

This will result in two files in the data/openwebtext/ folder, named train.bin (containing ~9B tokens) and val.bin (containing ~4M tokens). Now, train a 1-layer transformer model with embedding dimension 128:
```
python train.py config/train_gpt2.py --wandb_project=monosemantic --n_layer=1 --n_embd=128 --n_head=4 --max_iters=200000 --lr_decay_iters=200000
```

This run saves the model checkpoints in the subfolder transformer/out. I trained the model for 200000 iterations in order to match the number of training epochs with Anthropic's paper. This run took around 3 days on an A100 GPU and achieved a validation loss of 4.609. 

If you have a node with more than one GPU available, you may alternatively train the model as follows for faster training. Here num_gpus is the number of GPUs on the node.

```
torchrun --standalone --nproc_per_node=num_gpus train.py config/train_gpt2.py --wandb_project=monosemantic --n_layer=1 --n_embd=128 --n_head=4 --max_iters=200000 --lr_decay_iters=200000
```

**step 2: generate training data for autoencoder**

Now move to the autoencoder subdirectory. 
```
cd ../autoencoder 
```

First, generate the training data for the autoencoder. 
```
python generate_mlp_data.py
```
By default, this computes MLP activations for 4 million contexts, and samples and randomly shuffles the outputs for 200 tokens per context. The dataset is saved in n_files=20 files in 'sae_data' subfolder of autoencoder. You may choose different values for these variables using --total_contexts, --tokens_per_context and --n_files command line arguments.

I used a node with 1TB RAM for this step as the dataset takes about 770GB space. I saved it in 20 files in order to be able to train the autoencoder model on a node with less CPU RAM (as low as 64GB) in Step 3.  

By default, MLP activations were saved in float16 data type, but you may change that by passing '--convert_to_f16=False' flag in the command line input. 

**step 2a: choose a subset of data for neuron resampling** 

Anthropic used a random subset of 819200 activation vectors to resample neurons four times during training. As the node that I used for training (in Step 3) did not have high enough RAM so that I could load the entire training data of the autoencoder and select 819200 examples at the time of resampling, I used a high-RAM (> 1TB) node to pre-select 4*819200 examples and saved it in a separate file 'data_for_resampling_neurons.pt'. 

This may be done as follows. 
```
python select_resampling_data.py 
```

If you have high-RAM available on your GPU node, you may skip this step and sample the subset randomly at the time of neuron resampling.

**step 3: train a sparse autoencoder model**

Next, you may train the sparse autoencoder model as follows. 
```
python train.py --l1_coeff=3e-7 
```

I tried a few different values of the L1-coefficient and learning rate and noticed that the best trade-off between feature activation sparsity (=L0-norm) and reconstructed NLL score occured around l1-coeff=3e-7 and learning_rate=3e-4. This L1 coefficient is much smaller than the values of L1-coefficient used in Anthropic's paper. I do not know why this is the case. 


## analysis of features
During training, I logged various metrics including feature density histograms. They are available on this [Weights & Biases project](https://wandb.ai/shehper/sparse-autoencoder-openwebtext-public). The spikes in various loss curves appear at the training step of neuron resampling, as one would expect. 

It is mentioned in the Anthropic paper that they performed manual inspection of features during training. I did not perform this manual inspection *during* training but I did perform it after training finished to compare different models. 

For this step, I used top_activations.py as
```
python top_activations.py --autoencoder_subdir=/subdirectory/of/out_autoencoder/containing_model_ckpt --eval_contexts=20000 --length_context_on_each_side=10 --k=10 --publish_html=True
```

where /subdirectory/of/out_autoencoder/containing_model_ckpt is the name of the subdirectory of 'out_autencoder' folder containing the model checkpoint. This evaluates the model on 20000 contexts from the OpenWebText dataset. The output is saved as a dictionary of k=10 top activations for each autoencoder neuron. If we pass publish_html=True, it also saves the top 10 activations and the associated tokens and contexts for each neuron in the form of an HTML file in the same subdirectory.

For example, please see the HTML files [high_density_neurons.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/high_density_neurons.html) and [ultra_low_density_neurons.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/ultra_low_density_neurons.html) for the model with l1_coeff=3e-7, learning_rate=3e-4, and loss curves as in the afore-mentioned [Weights & Biases page](https://wandb.ai/shehper/sparse-autoencoder-openwebtext-public/runs/rajo0rsx?workspace=user-).


## comments
**Preliminary observations.**

1. I found the algorithm suggested by Anthropic for neuron resampling to be extremely useful. With all of the other hyperparameters fixed (in particular, l1_coeff=3e-7, learning_rate=3e-4), I found that only ~200 neurons were alive without any neuron resampling, ~350 neurons were alive with a naive neuron resampling procedure (i.e. I re-initialized the weights of dead neurons using the PyTorch default Kaiming Uniform initialization), and 1700+ neurons were alive with Anthropic's neuron resampling procedure. 

2. High density neurons as seen in [high_density_neurons.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/high_density_neurons.html) were quite interpretable. I studied the first twenty neurons in this file, and wrote down the common theme about the context in which these neurons seem to fire. Some of the ultra-low density neurons were also interpretable. (See [ultra_low_density_features.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/ultra_low_density_neurons.html).) Perhaps, this is because my cutoff (1e-3 in the feature density histogram shown above) for separating high vs ultra-low density was not very precise. 

3. Almost 1/3rd of the ~1700 of neurons in high-density cluster were single-token features. (You may CTRL+F/CMD+F "single-token" in [high_density_neurons.html](https://shehper.github.io/monosemantic/autoencoder/out_autoencoder/1704783101.13-autoencoder-openwebtext/high_density_neurons.html) to look at these single-token features.)

4. When sorted by highest activation value, several of the top neurons seemed to single-token features. 

5. OpenWebText dataset is mostly monolingual, (I have not computed the exact percentage of English tokens in it yet) but the autoencoder learns features on texts of other languages. For example, the following neurons fire on specific languages or scripts. 3656: Spanish, 1305: Scandinavian, 77: South Slavic languages, 1682: Greek script, and 2601 and 215: Cyrilic. 

6. Quite interestingly, neuron # 215 seems to fire on Cyrilic vowels е, у, и, а, while neuron # 2601 seems to be active on Cyrilic consonants л, р, в, т.  

7. I see several � tokens in top activations of neurons. I don't know what these tokens mean, but perhaps this indicates that the OpenWebText contains some characters that BytePairEncoding does not encode. I intend to investigate this further.

**Possible improvements**
1. **Data type of training data.** I saved MLP activations in float16 instead of float32 in order to save twice the number of examples in the same storage space at the cost of precision in each value. 
2. **Size of training data.** I used only 800M activation vectors; in contrast, Anthropic used 8B activation vectors. I made this choice and the previous choice because of storage constraints, though I hope to be able to obtain more storage soon.  
3. **Dead and ultra-low density neurons.** Out of 4096 neurons, I found around 2324 to be dead, and only around 17 to be in the ultra-low density cluster. I expect that with a combination of more training data (as I trained on only 1/10th of the number of training examples used by Anthropic), float32 data type and more hyperparameter-tuning, the number of dead neurons will decrease significantly and the number of ultra-low density neurons will increase.
4. **Number of evaluation tokens**. I evaluated my autoencoder on only 2e5 tokens while Anthropic evaluated their autoencoder on 1e7 tokens. Perhaps, by increasing the number of evaluation tokens, I will see more ultra-low density neurons and less dead neurons.
5. **A more complete analysis of features**. While the top 10 activations of most features seem to show clear patterns about contexts where these features are active, a more detailed analysis as done by Anthropic in their sections on [Detailed Analysis of Individual Features](https://transformer-circuits.pub/2023/monosemantic-features/index.html#feature-analysis) and [Global Analysis](https://transformer-circuits.pub/2023/monosemantic-features/index.html#global-analysis) needs to be done. 
6. **Feature ablations**. I must ablate features to see their causal effect on next-token-prediction. 

**Some curiosities**
1. *L1-coefficient.* I do not know why the L1-coefficient for my training runs was three-to-four orders of magnitude smaller than the values used by Anthropic. (I had 3e-7, while they had values of the order of 1e-3 or 1e-4.)

2. *GeLU vs ReLU activations.* nanoGPT uses GeLU activations in its MLP layer. In contrast, Anthropic used ReLU activations in its MLP layer. I wonder if this small but subtle difference has a significant effect on the kind of features learned by the sparse autoencoder.


