import os
from unittest import skip
import pandas as pd
import ipdb
from torch.utils.data import DataLoader
from torch import nn
import torch
from data import QGDataset
from utils import *
from eval import eval_ceaf
from datasets import load_metric

from transformers.models.led import LEDConfig, LEDTokenizer, LEDForConditionalGeneration
import pytorch_lightning as pl
from clearml import StorageManager, Dataset as ClearML_Dataset


class LongformerQG(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()

        clearml_data_object = ClearML_Dataset.get(dataset_name=self.cfg.clearml_dataset_name, dataset_project=self.cfg.clearml_dataset_project_name,
                                                  dataset_tags=list(self.cfg.clearml_dataset_tags), only_published=True)
        self.dataset_path = clearml_data_object.get_local_copy()

        print("CUDA available: ", torch.cuda.is_available())

        # Load and update config then load a pretrained LEDForConditionalGeneration
        self.base_model_config = LEDConfig.from_pretrained(
            self.cfg.model_name)

        # self.base_model_config.gradient_checkpointing = True
        # self.base_model_config.max_length = cfg.max_input_len
        # self.base_model_config.min_length = 24

        # Load tokenizer and metric
        self.tokenizer = LEDTokenizer.from_pretrained(
            self.cfg.model_name, use_fast=True)
        self.base_model = LEDForConditionalGeneration.from_pretrained(
            self.cfg.model_name, config=self.base_model_config)

        self.bleu_metric = load_metric('bleu')
        self.rouge_metric = load_metric('rouge')

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the self.task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).

        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        # global_attention_mask[:, :1] = 1
        global_attention_mask[(input_ids == self.tokenizer.cls_token_id)] = 1
        global_attention_mask[(input_ids == self.tokenizer.sep_token_id)] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        # Sets the questions for global attention
        # batch_size = input_ids.size()[0]
        # question_separators = (input_ids == 2).nonzero(as_tuple=True)
        # sep_indices_batch = [torch.masked_select(question_separators[1], torch.eq(
        #     question_separators[0], batch_num))[0] for batch_num in range(batch_size)]

        # for batch_num in range(batch_size):
        #     global_attention_mask[batch_num, :sep_indices_batch[batch_num]] = 1

        return global_attention_mask

    def forward(self, **batch):

        input_ids, attention_mask, question_ids = batch[
            "input_ids"], batch["attention_mask"], batch["question_ids"]

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,  # mask padding tokens
            global_attention_mask=self._set_global_attention_mask(
                input_ids),
            # decoder_input_ids=question_ids,
            labels=question_ids,
            output_hidden_states=True,
        )

        return outputs

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        if self.cfg.batch_size == 1:
            batch["input_ids"], batch["attention_mask"], batch["question_ids"] = batch[
                "input_ids"].unsqueeze(0), batch["attention_mask"].unsqueeze(0), batch["question_ids"].unsqueeze(0)
            outputs = self.forward(**batch)
        else:
            outputs = self.forward(**batch)

        return {'loss': outputs.loss}

    def training_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss)/len(total_loss))

    def _get_dataloader(self, split_name):
        """Get training and validation dataloaders"""

        context = pd.read_csv(os.path.join(
            self.dataset_path, "processed/para-{}.txt".format(split_name)), header=None, names=["context"], sep='\t', dtype=str)
        answer = pd.read_csv(os.path.join(
            self.dataset_path, "processed/src-{}.txt".format(split_name)), header=None, names=["answer"], sep='\t', dtype=str)
        question = pd.read_csv(os.path.join(
            self.dataset_path, "processed/tgt-{}.txt".format(split_name)), header=None, names=["question"], sep='\t', dtype=str)
        dataset_split = pd.concat([context, answer, question], axis=1)

        if self.cfg.debug:
            dataset_split = dataset_split[:10]

        dataset = QGDataset(dataset=dataset_split,
                            tokenizer=self.tokenizer, cfg=self.cfg)

        if split_name in ["dev", "test"]:
            return DataLoader(dataset, batch_size=self.cfg.eval_batch_size, num_workers=self.cfg.num_workers, collate_fn=QGDataset.collate_fn)
        else:
            return DataLoader(dataset, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, collate_fn=QGDataset.collate_fn)

    def train_dataloader(self):
        print("Loading train dataset")
        return self._get_dataloader('train')

    def val_dataloader(self):
        print("Loading dev dataset")
        return self._get_dataloader('dev')

    def test_dataloader(self):
        print("Loading test dataset")
        return self._get_dataloader('test')

    def generate(self, **batch):
        return self.base_model.generate(
            **batch, num_beams=5,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    def _evaluation_step(self, split, batch, batch_nb):
        if self.cfg.eval_batch_size == 1:
            batch["input_ids"], batch["attention_mask"], batch["question_ids"] = batch[
                "input_ids"].unsqueeze(0), batch["attention_mask"].unsqueeze(0), batch["question_ids"].unsqueeze(0)

        loss = self.forward(**batch).loss
        question_ids = batch.pop("question_ids", None)
        outputs = self.generate(**batch)
        generated_outcome = self.tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=True)
        gold = self.tokenizer.batch_decode(
            question_ids, skip_special_tokens=True)

        # results = self.bleu_metric.compute(
        #     predictions=generated_outcome, references=gold)

        results = self.rouge_metric.compute(
            predictions=generated_outcome, references=gold)

        # self.clearml_logger.report_scalar(
        #     title="batch_rouge_{}".format(split), series=split, value=results["rouge1"], iteration=batch_nb
        # )

        return loss, generated_outcome, results

    #################################################################################

    def validation_step(self, batch, batch_nb):
        batch_loss, batch_generated_text, batch_rouge = self._evaluation_step(
            'val', batch, batch_nb)
        return {"results": batch_rouge, "loss": batch_loss, "generated_text": batch_generated_text}

    def validation_epoch_end(self, outputs):
        total_loss = []
        total_rouge = []
        for batch in outputs:
            total_loss.append(batch["loss"])
            total_rouge.append(batch["results"]["rouge1"].mid.fmeasure)
        self.log("val_loss", sum(total_loss)/len(total_loss), )
        self.log("average_val_rouge1", sum(total_rouge)/len(total_rouge),)

    def test_step(self, batch, batch_nb):
        batch_loss, batch_generated_text, batch_rouge = self._evaluation_step(
            'test', batch, batch_nb)
        return {"results": batch_rouge, "loss": batch_loss, "generated_text": batch_generated_text}

    def test_epoch_end(self, outputs):
        total_loss = []
        total_rouge = []
        for batch in outputs:
            total_loss.append(batch["loss"])
            total_rouge.append(batch["results"]["rouge1"].mid.fmeasure)
        self.log("test_loss", sum(total_loss)/len(total_loss))
        self.log("average_test_rouge1", sum(total_rouge)/len(total_rouge))

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""

        # Freeze the model layers
        # for idx, (name, parameters) in enumerate(self.base_model.named_parameters()):
        #     if idx<6:
        #         parameters.requires_grad=False
        #     else:
        #         parameters.requires_grad=True

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return [optimizer]
