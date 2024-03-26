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