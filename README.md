## License Plate Recognition

### Built using combination of:
- Yolov5, trained on my local machine with the Large-License-Plate-Detection-Dataset. Link: https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset/data
- Fast plate OCR, a pretrained license plate OCR library by Andrés. Link: https://ankandrew.github.io/fast-plate-ocr/latest/
- Layers of preprocessing and other code added by me
- Initially run on a WSL Ubuntu Linux environment.

### Running the model:
- Current code is run on Python 3.12.7. But I think 3.12.x should work fine.
- Install requirements from requirements.txt.
- Install other system wide packages from your package manager.
- Run "python LPR6.py"
- Output will be produced in /validation_results
- train_yolov.py can be used to train the model on the dataset. Directory and file locations may need to be modified.
