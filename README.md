
# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

This repository reproduces results of [Anthropic's Sparse Dictionary Learning paper](https://transformer-circuits.pub/2023/monosemantic-features/). The codebase is quite rough, but the results are excellent. See the [feature interface](https://shehper.github.io/feature-interface/) to browse through the features learned by the sparse autoencoder.  There are improvements to be made (see the [TODOs](#todos) section below), and I will work on them intermittently as I juggle things in life :)

I trained a 1-layer transformer model from scratch using [nanoGPT](https://github.com/karpathy/nanoGPT) with $d_{\text{model}} = 128$. Then, I trained a sparse autoencoder with $4096$ features on its MLP activations as in [Anthropic's paper](https://transformer-circuits.pub/2023/monosemantic-features/). 93% of the autoencoder neurons were alive, only 5% of which were of ultra-low density. There are several interesting features. For example, there is [a feature for French language](https://shehper.github.io/feature-interface/?page=2011),

<p align="center">
  <img src="./assets/french.png" width="700" />
</p>

a feature each for German, Japanese, and many other languages, as well many other interesting features:

- [A feature for German](https://shehper.github.io/feature-interface/?page=156)
- [A feature for Scandinavian languages](https://shehper.github.io/feature-interface/?page=1634)
- [A feature for Japanese](https://shehper.github.io/feature-interface/?page=1989)
- [A feature for Hebrew](https://shehper.github.io/feature-interface/?page=2026)
- [A feature for Cyrilic vowels](https://shehper.github.io/feature-interface/?page=3987)
- [A feature for token "at" in words like "Croatian", "Scat", "Hayat", etc](https://shehper.github.io/feature-interface/?page=1662)
- [A single token feature for "much"](https://shehper.github.io/feature-interface/?page=2760)
- [A feature for sports leagues: NHL, NBA, etc](https://shehper.github.io/feature-interface/?page=379)
- [A feature for Gregorian calendar dates](https://shehper.github.io/feature-interface/?page=344)
- [A feature for "when"](https://shehper.github.io/feature-interface/?page=2022):
      - this feature particularly stands out because of the size of the mode around large activation values. 
- [A feature for "&"](https://shehper.github.io/feature-interface/?page=1916)
- [A feature for ")"](https://shehper.github.io/feature-interface/?page=1917)
- [A feature for "v" in URLs like "com/watch?v=SiN8](https://shehper.github.io/feature-interface/?page=27)
- [A feature for programming code](https://shehper.github.io/feature-interface/?page=45)
- [A feature for Donald Trump](https://shehper.github.io/feature-interface/?page=292)
- [A feature for LaTeX](https://shehper.github.io/feature-interface/?page=538)

<!-- - [Bigram feature 1?](https://shehper.github.io/feature-interface/?page=446)
[Bigram feature 2?](https://shehper.github.io/feature-interface/?page=482) -->

<!-- - [A feature for some negative words/news](https://shehper.github.io/feature-interface/?page=218) -->

### Training Details

I used the "OpenWebText" dataset to train the transformer model, to generate the MLP activations dataset for the autoencoder, and to generate the feature interface visualizations. The transformer model had $d_{\text{model}}= 128$, $d_{\text{MLP}} = 512$, and $n_{\text{head}}= 4$. I trained this model for $2 \times 10^5$ iterations to roughly match the number of epochs with [Anthropic's training procedure](https://transformer-circuits.pub/2023/monosemantic-features#appendix-transformer).

I collected the dataset of 4B MLP activations by performing forward pass on 20M prompts (each of length 1024), keeping 200 activation vectors from each prompt. Next, I trained the autoencoder for approximately $5 \times 10^5$ training steps at batch size 8192 and learning rate $3 \times 10^{-4}$. I performed neuron resampling 4 times during training at training steps $2.5 \times i \times 10^4$ for $i=1, 2, 3, 4$. See a complete log of the training run on the [W&B page](https://wandb.ai/shehper/sparse-autoencoder-openwebtext-public/runs/vjbcwjsf?nw=nwusershehper). The L1-coefficient for this training run is $10^{-3}$. I selected the L1-coefficient and the learning rate by performing a grid search.

For the most part, I followed the training procedure described in the [appendix](https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder) of Anthropic's original paper. I did not follow the improvements they suggested in their [January](https://transformer-circuits.pub/2024/jan-update/index.html) and [February](https://transformer-circuits.pub/2024/feb-update/index.html) updates. 

### TODOs
- Incorporate the effects of feature ablations in the feature interface. 
- Implement an interface to see "Feature Activations on Example Texts" as done by Anthropic [here](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1-math.html).
- Modify the code so that one can train a sparse autoencoder on activations of any MLP / attention layer.

### Related Work
There are several other very interesting works on the web exploring sparse dictionary learning. Here is a small subset of them.

- [Sparse Autoencoders Find Highly Interpretable Features in Language Models by Cunningham, et al.](https://arxiv.org/abs/2309.08600)
- [Sparse Autoencoders Work on Attention Layer Outputs by Kissane, et al.](https://www.lesswrong.com/posts/DtdzGwFh9dCfsekZZ/sparse-autoencoders-work-on-attention-layer-outputs)
- [Joseph Bloom's SAE codebase](https://github.com/jbloomAus/mats_sae_training) along with a blogpost on [trained SAEs for all residual stream layers of GPT-2 small](https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream) 
- [Neel Nanda's SAE codebase](https://github.com/neelnanda-io/1L-Sparse-Autoencoder) along with a [blogpost](https://www.lesswrong.com/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s)
- [Callum McDougall's exercises on SAEs](https://github.com/callummcdougall/sae-exercises-mats/tree/main)
- [SAE library by AI Safey Foundation](https://github.com/ai-safety-foundation/sparse_autoencoder)

