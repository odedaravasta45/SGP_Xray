MODEL WEIGHTS
=============
This folder expects the trained checkpoint:

    detector_best.pt

Copy it here after training the supplied sgp-cnn.ipynb:

    /kaggle/working/checkpoints/detector_best.pt

The notebook itself was uploaded without .pt model weights, so the ZIP cannot include the learned parameters. The Django app will show a clear setup message until this file is present.

The checkpoint must match the DenseNet121 + custom FPN + Faster R-CNN architecture in ml/model.py.
