# Perceptual quality metrics for TensorFlow

This project contains differentiable perceptual quality metrics implemented in
the TensorFlow framework.

## Installation

```bash
pip install https://github.com/google-research/perceptual-quality/archive/master.zip
```

## PIM

This is an implementation of the perceptual information metric, as described in:

> "An Unsupervised Information-Theoretic Perceptual Quality Metric"<br />
> S. Bhardwaj, I. Fischer, J. Ballé, T. Chinen<br />
> https://proceedings.neurips.cc/paper/2020/file/00482b9bed15a272730fcb590ffebddd-Paper.pdf

Usage:

```python
from perceptual_quality import pim

model = pim.load_trained("pim-5")

# image_A, image_B: 4D tensors, batch x height x width x 3, sRGB colorspace.
# Returns the PIM distance between A and B for each batch element.
distance = model(image_A/255, image_B/255)

# Returns a `Distribution` object with the latent representation of the image.
distribution = model(image_A/255)
```

Refer to the online help of `pim.PIM.call()` for further information on
supported image formats, etc.

## NLPD

This is an implementation of the normalized Laplacian pyramid distance, as
described in:

> "Perceptually optimized image rendering"</br>
> V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
> https://doi.org/10.1364/JOSAA.34.001511

Usage:

```python
from perceptual_quality import nlpd

# image_A, image_B: at least 3D tensors, height x width x 3, sRGB or grayscale.
# Will return one number per color channel. Can also be batched.
distance = nlpd.nlpd(image_A, image_B)
distance = nlpd.nlpd_fast(image_A, image_B)

# Returns a list of tensors with the subband representation of the image.
model = nlpd.NLP()
subbands = model(image_A)
```

Refer to the online help of `nlpd.nlpd()`, `nlpd.nlpd_fast()`, and `nlpd.NLP()`
for further information on supported image formats, assumed display
characteristics, etc.

## Authors

* Sangnie Bhardwaj (github: [sangnie](https://github.com/sangnie))
* Johannes Ballé (github: [jonycgn](https://github.com/jonycgn))
* Ian Fischer (github: [iansf](https://github.com/ssjhv))

Note that this is not an officially supported Google product.
