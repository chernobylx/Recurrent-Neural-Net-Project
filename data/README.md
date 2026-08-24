# Data

This project uses the [Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started/data) competition dataset. Per the competition rules, the data is not committed to this repository.

To run the notebook locally, place `train.csv` and `test.csv` in this directory:

```bash
kaggle competitions download -c nlp-getting-started -p data/
unzip data/nlp-getting-started.zip -d data/
```

(Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and accepting the competition rules on Kaggle.)

| File | Rows | Description |
|---|---|---|
| `train.csv` | 7,613 | Tweets with `id`, `text`, `keyword`, `location`, and binary `target` (1 = real disaster) |
| `test.csv` | 3,263 | Same columns minus `target` |
