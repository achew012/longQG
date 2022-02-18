import torch
from torch.utils.data import Dataset
import ipdb

role_map = {
    'PerpOrg': 'perpetrator organizations',
    'PerpInd': 'perpetrator individuals',
    'Victim': 'victims',
    'Target': 'targets',
    'Weapon': 'weapons'
}


class QGDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset, tokenizer, cfg):
        self.tokenizer = tokenizer

        self.processed_dataset = {
            "docid": [],
            "context": [],
            "input_ids": [],
            "attention_mask": [],
            "qns": [],
            "start": [],
            "end": []
        }

        dataset["context_answer"] = dataset["answer"] + \
            " {} ".format(self.tokenizer.sep_token)+dataset["context"]

        self.context_answer_ids = [
            self.tokenizer(
                row["context_answer"], padding="max_length", truncation=True,
                max_length=cfg.max_input_len, return_tensors="pt")
            for idx, row in dataset.iterrows()]

        self.question_ids = [
            self.tokenizer(row["question"], padding="max_length",
                           truncation=True, max_length=cfg.max_output_len, return_tensors="pt")["input_ids"]
            for idx, row in dataset.iterrows()]

        # ipdb.set_trace()

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.question_ids)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        item = {}
        item['input_ids'] = self.context_answer_ids[idx]["input_ids"]
        item['attention_mask'] = self.context_answer_ids[idx]["attention_mask"]
        item['question_ids'] = self.question_ids[idx]
        return item

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        input_ids = torch.stack([ex['input_ids'] for ex in batch]).squeeze()
        attention_mask = torch.stack(
            [ex['attention_mask'] for ex in batch]).squeeze()
        question_ids = torch.stack([ex['question_ids']
                                   for ex in batch]).squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'question_ids': question_ids
        }
