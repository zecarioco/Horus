import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            k: torch.tensor(v[idx], dtype=torch.long)
            for k, v in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class DataManager:
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_tupye(self, split_train="train", split_test="test"):
        ds_train = load_dataset("Silly-Machine/TuPyE-Dataset", "multilabel", split=split_train)
        ds_test = load_dataset("Silly-Machine/TuPyE-Dataset", "multilabel", split=split_test)
        return ds_train, ds_test

    def tokenize(self, df, text_col="text", max_length=128):
        return self.tokenizer(
            df[text_col].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )

    def run(self, max_length=128):
        ds_train, ds_test = self.load_tupye()

        train_df = ds_train.to_pandas()
        test_df = ds_test.to_pandas()

        label_cols = []
        for col in train_df.columns:
            if col == "text":
                continue

            values = set(train_df[col].unique())

            if values.issubset({0, 1}):
                label_cols.append(col)

        label_cols = sorted(label_cols)

        def row_to_list(row):
            return [int(row[c]) for c in label_cols]

        train_labels = train_df.apply(row_to_list, axis=1).tolist()
        test_labels = test_df.apply(row_to_list, axis=1).tolist()

        train_enc = self.tokenize(train_df, "text", max_length)
        test_enc = self.tokenize(test_df, "text", max_length)

        train_dataset = MultilabelDataset(train_enc, train_labels)
        test_dataset = MultilabelDataset(test_enc, test_labels)

        return {
            "train_df": train_df,
            "test_df": test_df,
            "label_cols": label_cols,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "tokenizer": self.tokenizer,
        }