---
layout: post
title:  "Vision transformer (VIT)"
date:   2021-05-11 12:10:41 +0430
categories: Transformers
permalink: /:categories/:title
author: 'Mohammad Dehghan'
---
1. Overview
  * Some notes from the paper
2. PyTorch impelementation
  * Patch embeddings
  * Positional encodings with parameters
  * MLP Classification Head
  * Vision Transformer


## Overview

I will describe the PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf) step by step in this blog .

Without any convolution layers, the Vision Transformer applies a pure transformer to images. The image is split into patches and a transformer is applied to patch embeddings. Patch embeddings are created by performing a simple linear transformation on the patch's flattened pixel values. TThe patch embeddings and a classification token are then fed into a standard transformer encoder. With an MLP, the image is classified by the encoding of the classification token.

Because patch embeddings do not contain any information about where the patch came from, learned positional embeddings are added to the patch embeddings when feeding the transformer with patches. The positional embeddings, along with other parameters, are a set of vectors for each patch location..

# Some notes from the paper
* ViTs that are pre-trained on large datasets perform well. A single linear layer should be used when fine-tuning a model trained with an MLP classification head. By using a ViT pretrained on 300 million images, the paper beats SOTA. 
* While the patch size remains the same, they use higher resolution images during inference. In order to calculate positional embeddings for new patches, learning positional embeddings are interpolated.

## Implementation
# Patch embeddings
The paper divides the image into equal-sized patches and performs a linear transformation on each patch's flattened pixels.
Because it's easier to build, we employ a convolution layer to achieve the same result.

{% highlight ruby %}
class PatchEmbeddings(Module):
  def __init__(self, d_model: int, patch_size: int, in_channels: int):
   super().__init__()
{% endhighlight %}

* d_model is the transformer embeddings size
* patch_size is the size of the patch
* in_channels is the number of channels in the input image (3 for rgb)

A convolution layer with a kernel size equal to patch size and a stride length equal to patch size is created. This is the same as dividing the image into patches and applying a linear transformation to each patch individually.

{% highlight ruby %}
  self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)
{% endhighlight %}

* Apply the defined convolution layer
* Get the shape for next usuage.
* Rearrange from [batch_size, channels, height, width] to shape [patches, batch_size, d_model]
* Finally, return the patch embeddings

{% highlight ruby %}
  def forward(self, x: torch.Tensor):
    x = self.conv(x)
    bs, c, h, w = x.shape
    x = x.permute(2, 3, 0, 1)
    x = x.view(h * w, bs, c)
    return x
{% endhighlight %}


# Positional encodings with parameters

Learned positional embeddings are added to patch embeddings in this way.

* d_model is the transformer embeddings size
* max_len is the maximum number of patches
* Positional embeddings for each location

{% highlight ruby %}
class LearnedPositionalEmbeddings(Module):
  def __init__(self, d_model: int, max_len: int = 5_000):
    super().__init__()
    self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
{% endhighlight %}

* Get the positional embeddings for the patches you've provided.
* Add to patch embeddings and return

{% highlight ruby %}
  def forward(self, x: torch.Tensor):
    pe = self.positional_encodings[x.shape[0]]
    return x + pe
{% endhighlight %}


# MLP Classification Head
This is the two-layer MLP head that uses classification token embedding to classify the image.

* d_model is the transformer embedding size
* n_hidden is the size of the hidden layer
* n_classes is the number of classes in the classification task

{% highlight ruby %}
class ClassificationHead(Module):
  def __init__(self, d_model: int, n_hidden: int, n_classes: int):
  super().__init__()
  self.linear1 = nn.Linear(d_model, n_hidden)
  self.act = nn.ReLU()
  self.linear2 = nn.Linear(n_hidden, n_classes)

  def forward(self, x: torch.Tensor):
    x = self.act(self.linear1(x))
    x = self.linear2(x)
    return x
{% endhighlight %}


# Vision Transformer

The patch embeddings, positional embeddings, transformer, and classification head are all combined in this.

* transformer_layer is a copy of a single transformer layer. We make copies of it to make the transformer with n_layers .
* n_layers is the number of transformer layers.
* patch_emb is the patch embeddings layer.
* pos_emb is the positional embeddings layer.
* classification is the classification head.

{% highlight ruby %}
class VisionTransformer(Module):
def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
               patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                classification: ClassificationHead):

  super().__init__()
  self.patch_emb = patch_emb
  self.pos_emb = pos_emb
  self.classification = classification
  self.transformer_layers = nn.Transformer(transformer_layer, n_layers)
  self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
  self.ln = nn.LayerNorm([transformer_layer.size])

  def forward(self, x: torch.Tensor):
    x = self.patch_emb(x)
    x = self.pos_emb(x)
    # Concatenate the classification token embeddings before feeding the transformer
    cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
    x = torch.cat([cls_token_emb, x])
    for layer in self.transformer_layers:
      x = layer(x=x, mask=None)
    x = x[0]
    x = self.ln(x)
    x = self.classification(x)
    return x
{% endhighlight %}