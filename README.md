Improving Native CNN Robustness with Filter Frequency Regularization
===============================================================================

*Jovita Lukasik\*, Paul Gavrikov\*, Janis Keuper, Margret Keuper*.\
TMLR 2023

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]


## Abstract 
Neural networks tend to overfit the training distribution and perform poorly on out-of-distribution data. A conceptually simple solution lies in adversarial training, which introduces worst-case perturbations into the training data and thus improves model generalization to some extent. However, it is only one ingredient towards generally more robust models and requires knowledge about the potential attacks or inference time data corruptions during model training. This paper focuses on the native robustness of models that can learn robust behavior directly from conventional training data without out-of-distribution examples. To this end, we investigate the frequencies present in learned convolution filters. Clean-trained models often prioritize high-frequency information, whereas adversarial training enforces models to shift the focus to low-frequency details during training. By mimicking this behavior through frequency regularization in learned convolution weights, we achieve improved native robustness to adversarial attacks, common corruptions, and other out-of-distribution tests. Additionally, this method leads to more favorable shifts in decision-making towards low-frequency information, such as shapes, which inherently aligns more closely with human vision.

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


## Installation 
In order to use this code, install requirements:
```bash
pip install -r requirements.txt
```

## Reproduce our results

- Example run with EfficientNet-B0 without any modified layers
```
python train.py --data_dir  DATA_DIR --output_dir OUT_DIR --model efficientnet-b0  \
        --basis_filter None --scheduler Cosine --learning_rate 0.001 --optimizer adamw \
        --weight_decay 0.05 --l2_reg 0 --l2_lambda 0.01 
```
- Example run with EfficientNet-B0 using DCT-based modifications {WD, SD} and regularization
```
python train.py --data_dir  DATA_DIR --output_dir OUT_DIR --model efficientnet-b0  \
        --basis_filter WD --scheduler scheduler --learning_rate 0.001 --optimizer adamw \
        --weight_decay 0.05 --l2_reg 1 --l2_lambda 0.01 
```

- Example run with ResNet-9 using DCT-based modifications {WD, SD} and regularization
```
python train.py --data_dir  DATA_DIR --output_dir OUT_DIR --model lowres_resnet9  \
        --basis_filter WD --scheduler Step --learning_rate 0.01 --optimizer adamw \
        --weight_decay 0.01 --l2_reg 1 --l2_lambda 0.01 
```

## Citation 

If you find our work useful in your research, please consider citing:

```bash
@article{LGKK23,
  author    = {Jovita Lukasik and
               Paul Gavrikov and
               Janis Keuper and
               Margret Keuper},
  title     = {Improving Native CNN Robustness with Filter Frequency Regularization},
  journal   = {Transactions on Machine Learning Research},
  url       = {https://openreview.net/forum?id=2wecNCpZ7Y},
  year      = {2023},
}
```

### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].
