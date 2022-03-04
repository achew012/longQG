import hydra
from omegaconf import OmegaConf
from typing import Dict, Any
import ast
import os
from model import LongformerQG
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task, StorageManager

Task.add_requirements(package_name="datasets", package_version="1.18.3")
Task.add_requirements("rouge_score")
Task.add_requirements("nltk")


def get_clearml_params(task: Task) -> Dict[str, Any]:
    """
    returns task params as a dictionary
    the values are casted in the required Python type
    """
    string_params = task.get_parameters_as_dict()
    clean_params = {}
    for k, v in string_params["General"].items():
        try:
            # ast.literal eval cannot read empty strings + actual strings
            # i.e. ast.literal_eval("True") -> True, ast.literal_eval("i am cute") -> error
            clean_params[k] = ast.literal_eval(v)
        except:
            # if exception is triggered, it's an actual string, or empty string
            clean_params[k] = v
    return OmegaConf.create(clean_params)


def train(cfg, task) -> LongformerQG:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./",
        filename="best_ner_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        period=5,
    )

    model = LongformerQG(cfg, task)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    return model


def test(cfg, model) -> list:
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.num_epochs)
    results = trainer.test(model)
    return results


@hydra.main(config_path=os.path.join("..", "config"), config_name="config")
def hydra_main(cfg) -> float:

    print("Detected config file, initiating task... {}".format(cfg))

    if cfg.train:
        task = Task.init(
            project_name="LongQG",
            task_name="longQG-train",
            output_uri="s3://experiment-logging/storage/",
        )
    else:
        task = Task.init(
            project_name="LongQG",
            task_name="longQG-predict",
            output_uri="s3://experiment-logging/storage/",
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    task.set_base_docker("nvidia/cuda:11.2.0-runtime-ubuntu20.04")
    task.execute_remotely(queue_name="compute", exit_process=True)

    cfg = get_clearml_params(task)

    if cfg.train:
        model = train(cfg, task)

    if cfg.test:
        if cfg.trained_model_path:
            trained_model_path = StorageManager.get_local_copy(cfg.trained_model_path)
            model = LongformerQG.load_from_checkpoint(
                trained_model_path, cfg=cfg, task=task
            )

        results = test(cfg, model)


if __name__ == "__main__":
    hydra_main()
