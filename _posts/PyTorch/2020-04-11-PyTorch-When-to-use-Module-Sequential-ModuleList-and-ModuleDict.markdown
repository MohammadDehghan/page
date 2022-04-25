---
layout: post
title:  "PyTorch: When to use Module, Sequential, ModuleList, and ModuleDict"
date:   2020-04-11 11:00:36 +0430
categories: PyTorch
permalink: /:categories/:title
author: 'Mohammad Dehghan'
---

We'll look at how to use PyTorch's three primary building blocks: Module, Sequential, and ModuleList in this blog. We'll start with an example and improve it incrementally.

torch.nn contains all three of these classes.

{% highlight ruby %}
import torch.nn as nn

nn.Module
nn.Sequential
nn.Module
{% endhighlight %}

## Module: the main building block

The Module is the most important component; it defines the basic class for all neural networks, and it is mandatory that you subclass it.

As an example, let's make a classic CNN classifier:

{% highlight ruby %}
import torch.nn.functional as F

class MyCustomCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=50176, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=10, bias=True)
)
{% endhighlight %}

This is a very basic classifier with two layers of 3x3 convs + batchnorm + relu for the encoding and two linear layers for the decoding. You may have seen this type of scripting before if you aren't new to PyTorch, however there are two issues.

If we want to add a layer, we'll need to put a lot of code in the __init__ and forward functions once more. Also, if we wish to use a common block in another model, such as the 3x3 conv + batchnorm + relu, we must rewrite it.

## Sequential: stack and merge layers

Sequential is a container for Modules that can be stacked and executed simultaneously.

You'll notice that we have to store everything in self. Sequential can help us improve our code.

{% highlight ruby %}
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (conv_block1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (conv_block2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=50176, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
{% endhighlight %}

Isn't it a lot better now?

Have you seen how similar conv block1 and conv block2 are? To make the code even easier, we could construct a method that returns a nn.Sequential!

{% highlight ruby %}
def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )
{% endhighlight %}

Then, in our Module, we can simply call this function.

{% highlight ruby %}
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv_block1 = conv_block(in_c, 32, kernel_size=3, padding=1)
        
        self.conv_block2 = conv_block(32, 64, kernel_size=3, padding=1)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (conv_block1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (conv_block2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=50176, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
{% endhighlight %}


Even better! Even so, conv block1 and conv block2 are nearly identical! Using nn.Sequential, we can combine them. 

{% highlight ruby %}
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_c, 32, kernel_size=3, padding=1),
            conv_block(32, 64, kernel_size=3, padding=1)
        )

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (decoder): Sequential(
    (0): Linear(in_features=50176, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
{% endhighlight %}

The conv block is now held by self.encoder. Our model's logic has been decoupled to make it easier to read and reuse. The conv block function from one model can be imported and used in another.

## Dynamic Sequential: create multiple layers at once

What if we could add new layers to self.encoder instead of hardcoding them:

{% highlight ruby %}
self.encoder = nn.Sequential(
            conv_block(in_c, 32, kernel_size=3, padding=1),
            conv_block(32, 64, kernel_size=3, padding=1),
            conv_block(64, 128, kernel_size=3, padding=1),
            conv_block(128, 256, kernel_size=3, padding=1),

        )
{% endhighlight %}

Wouldn't it be wonderful if we could describe the sizes as an array and have the layers created automatically without having to write each one? We can, fortunately, make an array and feed it to Sequential.

{% highlight ruby %}
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.enc_sizes = [in_c, 32, 64]
        
        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (decoder): Sequential(
    (0): Linear(in_features=50176, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
{% endhighlight %}

Let's have a look at it in more detail. The sizes of our encoder are stored in an array called self.enc sizes. The sizes are then iterated to generate an array conv blocks. We ziped the size'array with itself by shifting it by one because we had to give booth an in size and an outsize for each layer.

Take a look at the following example to see what I mean:

{% highlight ruby %}
sizes = [1, 32, 64]

for in_f,out_f in zip(sizes, sizes[1:]):
    print(in_f,out_f)
{% endhighlight %}

{% highlight ruby %}
1 32
32 64
{% endhighlight %}

We then use the * operator to decompose it because Sequential does not take a list.

Awesome! We can now easily add a new number to the list if we just want to add a size. Making the size a parameter is a typical technique.

{% highlight ruby %}
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, enc_sizes, n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        
        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, [32,64, 128], 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (decoder): Sequential(
    (0): Linear(in_features=50176, out_features=1024, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=1024, out_features=10, bias=True)
  )
)
{% endhighlight %}

We can perform the same thing with the decoder.

{% highlight ruby %}
def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

class MyCustomCNN(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [64 * 28 * 28, *dec_sizes]

        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        dec_blocks = [dec_block(in_f, out_f) 
                       for in_f, out_f in zip(self.dec_sizes, self.dec_sizes[1:])]
        
        self.decoder = nn.Sequential(*dec_blocks)
        
        self.last = nn.Linear(self.dec_sizes[-1], n_classes)

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyCustomCNN(1, [32,64], [1024, 512], 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (decoder): Sequential(
    (0): Sequential(
      (0): Linear(in_features=50176, out_features=1024, bias=True)
      (1): Sigmoid()
    )
    (1): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): Sigmoid()
    )
  )
  (last): Linear(in_features=512, out_features=10, bias=True)
)
{% endhighlight %}


We followed the same method as before, creating a new decoding block, linear + sigmoid, and passing an array of sizes. A self had to be added. last, because we don't want to turn on the output

We can even divide our model into two parts now! Encoder + Decoder is a combination of the words encoder and decoder

{% highlight ruby %}
class MyEncoder(nn.Module):
    def __init__(self, enc_sizes):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])

        def forward(self, x):
            return self.conv_blocks(x)
        
class MyDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f) 
                       for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        return self.dec_blocks()
    
    
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [self.enc_sizes[-1] * 28 * 28, *dec_sizes]

        self.encoder = MyEncoder(self.enc_sizes)
        
        self.decoder = MyDecoder(self.dec_sizes, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.flatten(1) # flat
        
        x = self.decoder(x)
        
        return x
{% endhighlight %}


{% highlight ruby %}
model = MyCustomCNN(1, [32,64], [1024, 512], 10)
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): MyEncoder(
    (conv_blocks): Sequential(
      (0): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
  (decoder): MyDecoder(
    (dec_blocks): Sequential(
      (0): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
      )
      (1): Sequential(
        (0): Linear(in_features=1024, out_features=512, bias=True)
        (1): Sigmoid()
      )
    )
    (last): Linear(in_features=512, out_features=10, bias=True)
  )
)
{% endhighlight %}

It's worth noting that MyEncoder and MyDecoder could both yield a nn.Sequential. For models, I like the first design, and for building blocks, I prefer the second.

It is easier to communicate, debug, and test our module by breaking it down into submodules.

## ModuleList : when we need to iterate

ModuleList allows you to keep track of Modules in a list format. It can be handy in situations when you need to iterate through layers and store/use data, such as in U-net.

The key difference between Sequential and ModuleList is that the inner layers are not connected because ModuleList does not include a forward method. If we assume we require each layer's output in the decoder, we can store it as follows:

{% highlight ruby %}
class MyModule(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.trace = []
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            self.trace.append(x)
        return x
{% endhighlight %}

{% highlight ruby %}
model = MyModule([1, 16, 32])
import torch

model(torch.rand((4,1)))

[print(trace.shape) for trace in model.trace]
{% endhighlight %}

{% highlight ruby %}
torch.Size([4, 16])
torch.Size([4, 32])

[None, None]
{% endhighlight %}

## ModuleDict: when we need to choose

What if we want to change our conv block to LearkyRelu? ModuleDict can be used to generate a dictionary of Modules and dynamically switch Modules as needed.


{% highlight ruby %}
def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
    ])
    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )
{% endhighlight %}

{% highlight ruby %}
print(conv_block(1, 32,'lrelu', kernel_size=3, padding=1))
print(conv_block(1, 32,'relu', kernel_size=3, padding=1))
{% endhighlight %}

{% highlight ruby %}
Sequential(
  (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): LeakyReLU(negative_slope=0.01)
)
Sequential(
  (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
)
{% endhighlight %}

## Final implementation

Let's bring it all to a close!

{% highlight ruby %}
def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
    ])
    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )

def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

class MyEncoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1, *args, **kwargs) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])
        
        def forward(self, x):
            return self.conv_blocks(x)
        
class MyDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f) 
                       for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        return self.dec_blocks()
    
    
class MyCustomCNN(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes, activation='relu'):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [32 * 28 * 28, *dec_sizes]

        self.encoder = MyEncoder(self.enc_sizes, activation=activation)
        
        self.decoder = MyDecoder(dec_sizes, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.flatten(1) # flat
        
        x = self.decoder(x)
        
        return x

{% endhighlight %}


{% highlight ruby %}
model = MyCustomCNN(1, [32,64], [1024, 512], 10, activation='lrelu')
print(model)
{% endhighlight %}

{% highlight ruby %}
MyCustomCNN(
  (encoder): MyEncoder(
    (conv_blocks): Sequential(
      (0): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
      )
      (1): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.01)
      )
    )
  )
  (decoder): MyDecoder(
    (dec_blocks): Sequential(
      (0): Sequential(
        (0): Linear(in_features=1024, out_features=512, bias=True)
        (1): Sigmoid()
      )
    )
    (last): Linear(in_features=512, out_features=10, bias=True)
  )
)
{% endhighlight %}

## Conclusion
So, there you have it.

When you have a large block made up of several smaller blocks, use Module.
When you want to make a compact block out of layers, use Sequential.
When you need to loop through some levels or building blocks and accomplish anything, use ModuleList.
When you need to parametize some parts of your model, such as an activation function, use ModuleDict.
That's all there is to it, guys!

Thank you for taking the time to read this.