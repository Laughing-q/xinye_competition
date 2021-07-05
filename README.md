# xinye_competition

```plain
xinye_competition
├── data
│   ├── train
│   │   ├── a_images
│   │   ├── a_annotations.json
│   │   ├── b_images
│   │   ├── b_annotations.json
│   ├── test
│   │   ├── a_images
│   │   ├── a_annotations.json
│   │   ├── b_images
│   │   ├── b_annotations.json
│   ├── cropped_train
│   ├── cropped_test
│   ├── retail
│   │   ├── train
│   │   ├── val
├── model
├── model_files
├── submit
├── tools
├── utils
```

Supported methods:

- [  ] [CGD](https://arxiv.org/pdf/1903.10663v4.pdf)
- [  ] [CoAtNet](https://arxiv.org/pdf/2106.04803v1.pdf)

train and test folders are original data.

cropped_train/test and retail folders are generated by tools/prepare_data.py.

cropped_train for regressor training.

cropped_test for regressor testing.


retail for detector training.

```shell
cd xinye_competition
```

## Prepare data
```shell
python tools/prepare_data.py --regressor-data --detector-data --generate-pair
```

## Train regressor
```shell
python tools/train_regressor.py
```

## Train detector
```shell
python tools/train_detector.py
```

## Inference
```shell
python inference.py
```
