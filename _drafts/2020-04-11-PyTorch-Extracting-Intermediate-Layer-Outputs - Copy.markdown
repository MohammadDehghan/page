---
layout: post
title:  "PyTorch: Extracting Intermediate Layer Outputs"
date:   2020-04-11 11:00:36 +0430
categories: PyTorch
excerpt: "CIFAR-10 is a popular dataset small dataset for testing out Computer Vision Deep Learning learning methods. We're seeing a lot of improvements. But what is the human baseline?"
comments: true
permalink: /:categories/:title
author: 'Mohammad Dehghan'
---
1. Overview
2. Why are intermediate features necessary?
3. What is the best way to extract activations?
Preparations
Model
Feature extraction
4. Conclusion

<img src="/assets/Extracting Intermediate Layer Outputs-2.png" width="100%"> 
# Overview

We commonly work with predictions from the final layer of a neural network in deep learning tasks. In some circumstances, the outputs of intermediate layers may also be of relevance. It may be difficult to extract intermediate characteristics from the network, whether we wish to extract data embeddings or evaluate what prior layers have learned.
This blog post demonstrates how to use PyTorch's forward hook feature to harvest intermediate activations from any layer of a deep learning model. The simplicity and ability to extract features without having to run the inference twice is a significant benefit of this method, which only requires a single forward pass through the model to save many outputs.

# Why are intermediate features necessary?

Many applications benefit from the extraction of intermediate activations (also known as features). The outputs of intermediate CNN layers are commonly used to demonstrate the learning process and illustrate visual features discriminated by the model on different layers in computer vision challenges. Another common application is extracting intermediate outputs to build image or text embeddings, which can be used to detect duplicate items, include as input characteristics in a traditional ML model, show data clusters, and more. The outputs of intermediary layers can also be utilized to compress data into a smaller-sized vector carrying the data representation when dealing with Encoder-Decoder architectures. Intermediate activations can be useful in a variety of other situations. So let's talk about how to get them!


{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
