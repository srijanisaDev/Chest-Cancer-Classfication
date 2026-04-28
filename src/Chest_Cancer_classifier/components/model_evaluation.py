from __future__ import annotations

import json
from pathlib import Path

import mlflow
import tensorflow as tf

from Chest_Cancer_classifier import logger
from Chest_Cancer_classifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model, compile=False)
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    def _valid_generator(self):
        image_size = tuple(self.config.params_image_size[:-1])
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
        )

        self.valid_generator = data_generator.flow_from_directory(
            directory=self.config.training_data,
            target_size=image_size,
            batch_size=self.config.params_batch_size,
            subset="validation",
            shuffle=False,
            class_mode="categorical",
        )

    def _save_metrics(self, loss: float, accuracy: float):
        metrics = {"loss": float(loss), "accuracy": float(accuracy)}
        with open(self.config.metric_file_name, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logger.info(f"saved evaluation metrics to: {self.config.metric_file_name}")
        return metrics

    def _log_to_mlflow(self, metrics: dict):
        tracking_uri = self.config.mlflow_uri
        if "://" not in tracking_uri and not tracking_uri.startswith("file:"):
            tracking_uri = f"file:{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Chest_Cancer_Classification")

        with mlflow.start_run():
            mlflow.log_params({key: str(value) for key, value in self.config.all_params.items()})
            mlflow.log_param("model_path", str(self.config.path_of_model))
            mlflow.log_metric("loss", metrics["loss"])
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_artifact(str(self.config.metric_file_name))

    def evaluate(self):
        self._load_model()
        self._valid_generator()

        scores = self.model.evaluate(self.valid_generator, verbose=1)
        metrics = self._save_metrics(scores[0], scores[1])
        self._log_to_mlflow(metrics)
        return metrics