# Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering-Pytorch

#### *Working in Progress*

This repository corresponds to the ECCV 2020 paper *Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering*.

![](fig/overview.png)

## Dependencies

- Python 3.6
  - pytorch > 0.4
  - torchvision 0.2
  - h5py 2.7
  - tqdm 4.19

## Prepare Dataset (Follow [Cyanogenoid/vqa-counting](https://github.com/Cyanogenoid/vqa-counting))

- In the `data` directory, execute `./download.sh` to download VQA v2 and the bottom-up-top-down features.
- Prepare the data by running

```
python preprocess-features.py
python preprocess-vocab.py
```

This creates an `h5py` database (95 GiB) containing the object proposal features and a vocabulary for questions and answers at the locations specified in `config.py`.

## Training

### Step 1: Generating the paraphrases of questions



### Step 2: Adversarial training



## License

The code is released under the [MIT License](https://github.com/zaynmi/semantic-equivalent-da-for-vqa/blob/master/LICENSE)

## Citing

If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper: