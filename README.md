# Pytorch Implementation of Faster R-CNN
## Introduction
This repo is based on pytorch-1.1 and borrowed some code and techniques from maskrcnn-benchmark.
* **It is pure Pytorch code.** We convert all the numpy implementations to pytorch!
* **It supports multi-image batch training.** We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.
* **It supports multiple GPUs training.** We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.
* **It supports three pooling methods.** We integrate three pooling methods: roi pooing, roi align and roi crop. More importantly, we modify all of them to support multi-image batch training.
* **It is memory efficient.** We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. 