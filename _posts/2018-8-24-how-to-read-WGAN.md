# 解读：如何实现Wasserstein GAN

标签（空格分隔）： GAN

---

###### papers:
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

## Generative Adversarial Networks (GANs)
在解读WGAN（Wasserstein GAN）之前，必须先理解什么是GANs。GANs本质上是一种生成模型（generative model，简称G），它可以用于生成文本、图像、语音等任何“假”数据（fake data），为了提高“假”数据的质量，GANs引入另一个鉴别模型（discriminator，简称D），来分辨真实数据和“假”数据。生成模型和鉴别模型的持续对抗的过程中，两模型的造“假”和打“假”能力不断提升，“假”数据最终达到以假乱真的效果。这就是生成（Generative）和对抗（Adversarial）的由来。
#### GANs的实现过程：
- 创建两个独立模型：G和D。
- 为D创建loss函数，用于训练D区分真实数据和“假”数据。
- 为G创建loss函数，用于训练G生成出能够糊弄D的数据。

---
## Deep Convolutional Generative Adversarial Networks （DCGANs）
近年来，CNN广泛应用于有监督学习的机器视觉领域，相反地，CNN却很少被无监督学习领域所关注。DCGANs正是将CNN应用于GANs中的一种实现，WGAN也是建立在DCGANs之上的，所以在解读WGAN之前就需要理解DCGANs，它相比GANs模型的优点如下：
- 深度卷积神经网络在图像识别等机器视觉领域中的准确率远高于MLP。
- 可以将CNN的预训练结果迁移到GANs提高生成图像的准确率。GANs一直被诟病经常会生成和目标相差甚远的图像，而使用经过ImageNet等数据库预训练的CNN模型可以生成更接近真实图像的“假”图像。
- 通过滤波器（filter）可以形象化观测GANs如何生成图像。相比无法观测的MLP，可以通过观测filter来预估模型生成图像的效果。

#### 生成模型--Generator
![DCGANs.png](https://upload-images.jianshu.io/upload_images/13575947-e3cc14bfc8f09e38.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Generator是如何做到从一个长度为100的向量开始，最后得到一个rank 3的tensor的呢？

有经验的读者会发现，Generator就是一个转置的卷积神经网络，实际上它就是使用 deconvolution构建而成，它的功能和convolution刚好相反，用来增加feature map的grid size，所以Pytorch也将其称为transposed convolution。有关convolution和deconvolution的详细介绍，可以阅读[convolution arithmetic tutorial](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html)这篇博文。

和其他模型不同，Generator的input不是训练样本，而是一个噪声向量，由gen_noise()函数生成，通过generator（G）生成fake图像：
```
def gen_noise(bs):
  return V(torch.zeros((bs, nz, 1, 1)).normal_(0, 1))

fake = G(gen_noise(4))
fake_ims = md.trn_ds.denorm(fake)

fig, axes = plt.subplots(2, 2, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
  ax.imshow(fake_ims[i])
```
![image.png](https://upload-images.jianshu.io/upload_images/13575947-7d82926afffaeaad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
---
#### 鉴别模型--Discriminator
Discriminator实际上就是一个图像分类模型，和其他图像识别模型不同的是，它的输出是一个scalar，该值可以认为是真实数据（real data）和“假”数据（fake data）之间的差异，会被discriminator loss function用来优化Discriminator鉴别“假”数据的能力。

---
## Wasserstein GAN （WGAN）
理解了GANs背后的原理和DCGANs的网络架构，就可以来解读WGAN了。WGAN这篇[paper](https://arxiv.org/abs/1701.07875)很有意思，它里的所有数学公式都是跳过，直接通过如下图所示来实现：
![image.png](https://upload-images.jianshu.io/upload_images/13575947-8130d3edf6c7db91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这篇paper更了不起的是，它已经为你提供了能帮助理解模型背后的思想所需要的所有材料，如果你对其乃至GANs的原理感兴趣，那这一篇paper就足够了（你需要阅读paper里的大量的引用）。

这篇博文的目的是解读如何实现该神经网络，重点就是理解上述伪代码及其公式：
![1_9nGWityXFzNdgOxN15flRA.png](https://upload-images.jianshu.io/upload_images/13575947-369572234fbc453d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
while在这里体现的是对一个mini batch中所有real samples和fake samples的遍历，分别通过discriminator和generator的loss function来更新discriminator和generator的权值。要注意的是，while循环里有一个迭代次数为5（n_critic）的for循环，用以训练discriminator。由此也可以看出discriminator和generator的训练量比例为5：1，其原因是如果discriminator无法有效鉴别real data和fake data，那generator的“造假”能力即使再强也没有意义。

经过本文的解读，我相信你对WGAN、DCGANs和GANs已经有了一定了解，下篇博文我会解读如何用pytorch实现WGAN。

---
引用：
[Deep Learning 2: Part 2 Lesson 12](https://medium.com/@hiromi_suenaga/deep-learning-2-part-2-lesson-12-215dfbf04a94)












