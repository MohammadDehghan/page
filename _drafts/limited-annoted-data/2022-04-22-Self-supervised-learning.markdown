---
layout: post
title:  "Self-supervised learing Recent approaches"
date:   2022-04-22 11:00:36 +0430
categories: limited-annoted-data
permalink: /:categories/:title
author: 'Mohammad Dehghan'
---
* Do not remove this line (it will not be displayed) 
{:toc}


# Introduction

Unlabeled data is used in self-supervised learning to learn meaningful representations that can be used to a number of upstream tasks. VICReg is the most recent in a line of self-supervised image representation learning systems. SimCLR, MoCo, BYOL, and SimSiam are examples of notable antecedents, all of which aim to eliminate or replace something that was previously deemed to be necessary. VICReg continues in this spirit, offering a model that, on top of a simple invariance-preserving loss function, replaces all preceding gimmicks with two statistics-based regularization terms.

# Background

Traditional machine learning is largely supervised, which means in order to train such models annotating lots of annotated data are neccesary which is a time-consuming and expensive process.
In order to aliveate this problem, self-supervised learning generate those annotated data directly from input data. For instance, many recent language models such as BERT have been trained to guess missing words from raw text input. Consequently, The big advantage of self-supervised laening is that it enabels to train deep networks without requiring human annotated data.


<!-- ![my_pic]({{site.baseurl}}/assets/posts/SimCLR.png) -->

| ![space-1.jpg]({{site.baseurl}}/assets/posts/SimCLR.png){: width="500" } |
|:--:|
| <b>Augmented versions of the original image of a dog (a). [[Source]](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html).</b>|

{% highlight ruby %}
import torch.nn as nn

nn.Module
nn.Sequential
nn.Module
{% endhighlight %}

