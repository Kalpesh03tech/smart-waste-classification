# Training data

The training dataset is not included in this repository. To train the model, create a `dataset/` directory containing one subdirectory per class (`glass`, `metal`, `paper`, `plastic`) and place the corresponding images in each directory.

Example:

```text
dataset/
├── glass/
├── metal/
├── paper/
└── plastic/
```

Then run `python train.py --data_dir dataset/`.
