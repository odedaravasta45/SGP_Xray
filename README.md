# X-Ray AI Analyzer — Django + DenseNet121/FPN Faster R-CNN

A complete Django web application for uploading a chest X-ray, running the trained AMIA Public Challenge 2026 detector, drawing bounding boxes, showing finding labels and confidence, and downloading the annotated image.

## What the model does

The supplied notebook uses a custom **DenseNet121 + Feature Pyramid Network + Faster R-CNN** detector. It predicts 14 thoracic findings:

0. Aortic enlargement
1. Atelectasis
2. Calcification
3. Cardiomegaly
4. Consolidation
5. ILD
6. Infiltration
7. Lung Opacity
8. Nodule/Mass
9. Other lesion
10. Pleural effusion
11. Pleural thickening
12. Pneumothorax
13. Pulmonary fibrosis

Class 14 is the competition's **No finding** class; it is not a bounding-box disease class.

## Important: trained weights

Your uploaded `sgp-cnn.ipynb` contains the architecture/training/inference code, but the uploaded files do **not** contain `detector_best.pt`. Therefore the ZIP includes the complete Django/model integration, but not learned model weights.

After training the notebook, copy:

```text
/kaggle/working/checkpoints/detector_best.pt
```

to:

```text
models/detector_best.pt
```

Then restart Django.

## Windows setup

Recommended: Python 3.11 or 3.12 for the PyTorch/Django environment.

```powershell
cd xray_django_app
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py check
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## GPU

For NVIDIA GPU support, install the matching PyTorch CUDA build from the official PyTorch installer instructions instead of relying on the generic `torch` line in requirements.txt.

## Project structure

```text
xray_django_app/
├── manage.py
├── requirements.txt
├── README.md
├── models/
│   └── detector_best.pt       # add your trained weights here
├── ml/
│   ├── model.py               # DenseNet121 + FPN + Faster R-CNN
│   └── inference.py           # preprocessing, inference, NMS, boxes
├── detector/
│   ├── views.py               # upload + analysis endpoints
│   └── urls.py
├── xray_site/
├── templates/
├── static/
└── media/
```

## Medical-use note

This is an AI-assisted screening/research interface. It should not be presented as a definitive medical diagnosis. A qualified clinician should review the original X-ray and the model output.
