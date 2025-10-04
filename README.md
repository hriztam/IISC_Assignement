# IISC_Assignement

## How to train/test Vision Transformer

## Best Configs for my model

| Hyperparameter                           | Recommended Value                                 | Explanation                                                                                    |
| ---------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Image size**                           | 32×32 (native) or upscale to 48×48                | 32×32 works fine; 48×48 can help ViT learn better global features.                             |
| **Patch size**                           | 4×4                                               | This gives 64 patches per image (since 32/4 = 8), which is a good tradeoff for small datasets. |
| **Embedding dimension (d_model)**        | 256                                               | Keeps model capacity moderate.                                                                 |
| **Number of transformer layers (depth)** | 6                                                 | Enough for CIFAR-10 complexity; deeper models tend to overfit.                                 |
| **Number of attention heads**            | 4                                                 | 256 / 4 = 64-dimensional per head — standard ratio.                                            |
| **MLP hidden dim**                       | 512                                               | 2× the embedding size is typical.                                                              |
| **Dropout / Stochastic depth**           | 0.1–0.2                                           | Helps regularize on small datasets.                                                            |
| **Optimizer**                            | AdamW (lr=3e-4, weight_decay=0.05)                | Standard choice for ViT.                                                                       |
| **Scheduler**                            | Cosine decay with warmup (5 epochs)               | Smooth learning rate annealing.                                                                |
| **Batch size**                           | 128                                               | Works well with typical GPUs.                                                                  |
| **Training epochs**                      | 100–200                                           | ViT needs longer training than CNNs.                                                           |
| **Augmentations**                        | Mixup + CutMix + RandAugment                      | Essential for preventing overfitting.                                                          |
| **Normalization**                        | LayerNorm (inside transformer), ImageNet mean/std | Standard practice.                                                                             |

### If you want to train the model

1. Change the runtime -> Any GPU(ex: T4)
2. Change the model's config according to your needs (I optimized it for my colab GPU)
3. Run the notebook from cell 1

### If you want to test the model

1. Run all the cells till Cell 5 (Model Configuration)
2. Download the model checkpoint file ([Google Drive](https://drive.google.com/file/d/1a1xFMbNW7l00OgN5_1naxTvnV_YfRqMt/view?usp=sharing))
3. Run cell 12 (Make changes accordingly)

### Results Table

| Epoch | Training Loss | Accuracy |
| ----- | ------------- | -------- |
| 25    |    0.651      | 76.69%   |
| 50    |    0.28       | 81.94%   |
| 75    |    0.072      | 82.82%   |
| 100   |    0.021      | 83.9%    |

## How to test Text Segmentation Model

1. Change the runtime -> Any GPU(ex: T4)
2. Run the cells sequentially
3. After the setup is complete:
   - Run Cell 5 for automatic image segmentation demo
   - Run Cell 8 for automatic video segmentation demo
   - Uncomment functions in Cell 9 for interactive mode

Note: **Make sure you uncomment the functions when testing**

Example usage:

```py
image, boxes, labels, masks = segmentor.segment_image('image.jpg', 'cat')
frames, segments, labels = video_seg.segment_video('video.mp4', 'person')
```
