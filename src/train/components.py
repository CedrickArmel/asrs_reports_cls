"""Defines the components of the Train pipeline."""
import json
import os
from typing import Tuple

from dotenv import load_dotenv
import hopsworks
from fire import Fire
from hsfs.client.exceptions import RestAPIError
import keras
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import tensorflow as tf

from transformers import AutoTokenizer

from src.train.models import (
    CustomModelForSequenceClassification as Cls)
from src.utilitis.core import get_logger

logger = get_logger(__name__)

load_dotenv()

core = OmegaConf.load("conf/base/core.yaml")
etlconf = OmegaConf.load("conf/base/etl.yaml")
trainconf = OmegaConf.load("conf/base/train.yaml")

BATCH = trainconf.training.batch
EVAL_BATCH = trainconf.training.eval_batch
FG_NAME = core.feature_group.name
FG_VERSION = core.feature_group.version
FV_NAME = core.feature_view.name
FV_VERSION = core.feature_view.version
LABELS = etlconf.components.labels
METRICS = trainconf.model.metrics
MODEL_NAME = trainconf.model.name
NUM_CHECKPOINTS = trainconf.training.num_checkpoints
OPTIMIZER = trainconf.model.optimizer
AIP_TENSORBOARD = os.getenv("AIP_TENSORBOARD_LOG_DIR")
TENSORBOARD_HFREQ = trainconf.tb.hfreq  # 1
TENSORBOARD_UFREQ = trainconf.tb.ufreq  # 'epoch'
TD_DESCRIPTION = core.training_data.description
TD_FORMAT = core.training_data.format
TD_NEW = core.training_data.new
TD_VERSION = core.training_data.version
SEED = core.seed
STOP_POINT = trainconf.training.stop_points

project = hopsworks.login()
fs = project.get_feature_store()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Train(object):
    """Train pipeline components"""
    def __init__(self):
        self.strategy = tf.distribute.MirroredStrategy()

    def create_training_data(self, td_metadata_output: str):
        """Create a training data in the Hopsworks Feature Store.

        Args:
            td_metada_output (str): Path where to output \
                the training data metadata artefact.
        """
        logger.info("⏳ Starting Create Training Data task...🔄")

        try:
            fv = fs.get_feature_view(
                name=FV_NAME,
                version=FV_VERSION)
        except RestAPIError as e:
            logger.warning("⚠️ %s raised. Feature View seems missing. \
                           Trying to create a new one...🔄", str(e))
            try:
                fg = fs.get_feature_group(
                    name=FG_NAME,
                    version=FG_VERSION)
                # TODO: Externalise query as a module
                query = fg.select_all(include_primary_key=False)
                fv = fs.create_feature_view(
                    name=FV_NAME,
                    version=FV_VERSION,
                    query=query)
            except RestAPIError as e:
                logger.exception(
                    "💀 %s raised. Feature View creation failed ❌. \
                        Seems that Feature Group is also missing. \
                            A Feature Group is need before running \
                                this pipeline !", str(e))

        version, job = fv.create_train_validation_test_split(
            validation_size=0.3,
            test_size=0.2,
            description=TD_DESCRIPTION,
            seed=SEED,
            data_format=TD_FORMAT,
            write_options={"use_spark": True})
        metadata = json.loads(fv.json())

        if (version > TD_VERSION) and TD_NEW:
            logger.warning(
                "⚠️ Using the training data version %s as verion %s \
                    already exists AND 'td_new' parameter in conf \
                        have been set to 'True'. to change this behavior, \
                            set 'td_new' to 'false'.",
                str(version), str(TD_VERSION))
            metadata["training_data_version"] = version
        else:
            metadata["training_data_version"] = TD_VERSION

        if not os.path.exists(os.path.dirname(td_metadata_output)):
            os.makedirs(os.path.dirname(td_metadata_output))

        with open(td_metadata_output, 'w') as f:
            f.write(json.dumps(metadata))

        logger.info("⌛️ Create Training Data task completed successfully!✅")

    def _import_training_data(
            self, td_metadata_in: str) -> Tuple:
        """Import the training data from Hopsworks Feature Store.

        Args:
            td_metadata (str): Path of the training data metadata.

        Returns:
            Tuple[pd.DataFrame]: (Features Dataframe, Target DataFrame).
        """
        with open(td_metadata_in, 'r') as file:
            metadata = json.load(file)

        fv_name = metadata["name"]
        fv_version = metadata["version"]
        td_version = metadata["training_data_version"]
        fv = fs.get_feature_view(name=fv_name, version=fv_version)
        X_train, X_val, X_test, y_train, y_val, y_test = fv.\
            get_train_validation_test_split(
                training_dataset_version=td_version)
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        return train_data, val_data, test_data

    @tf.py_function(Tout=[tf.int32, tf.int16])
    def _tokenizer_fn(x, y):
        x = tf.compat.as_str(x.numpy()[0])
        x = tokenizer(
            x,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="tf")
        x = tf.convert_to_tensor(
            [v for v in x.values()])
        y = tf.convert_to_tensor(
            np.array(y, dtype=np.int16))
        return x, y

    def _import_model(self, state: str | None = None) -> Cls:
        loss = keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0.0,
            axis=-1,
            reduction="sum_over_batch_size",
            name="binary_crossentropy")
        if state != "no_checkpoint":
            logger.info("🏁 Importing model from checkpoint...🔄")
            with self.strategy.scope():
                model = keras.models.load_model(state)
        else:
            logger.info("🛠️ Initialising model from HuggingFace🤗...🔄")
            with self.strategy.scope():
                model = Cls.from_pretrained(MODEL_NAME,
                                            num_labels=len(LABELS),
                                            from_pt=True)
                model.compile(optimizer=OPTIMIZER,
                              loss=loss, metrics=METRICS)
        logger.info("⌛️ Model import completed successfully!✅")
        return model

    def _prepare_dataset(self, td_metadata_in: str):
        train_data, val_data, test_data = self._import_training_data(
            td_metadata_in=td_metadata_in)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ([item for item in train_data[0].values.tolist()],
             [item for item in train_data[1].values.tolist()]))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ([item for item in val_data[0].values.tolist()],
             [item for item in val_data[1].values.tolist()]))
        test_dataset = tf.data.Dataset.from_tensor_slices(
            ([item for item in test_data[0].values.tolist()],
             [item for item in test_data[1].values.tolist()]))

        # Below we repeat infinitely because we want to implement virtual
        # epochs and contant training sample size (as new data arrives)
        train_dataset = train_dataset.map(
            lambda x, y: self._tokenizer_fn(x, y),
            num_parallel_calls=tf.data.AUTOTUNE)\
            .shuffle(buffer_size=tf.data.AUTOTUNE,
                     seed=SEED)\
            .batch(BATCH)\
            .repeat()\
            .cache()\
            .prefetch(tf.data.AUTOTUNE)

        val_dataset = val_dataset.map(
            lambda x, y: self._tokenizer_fn(x, y),
            num_parallel_calls=tf.data.AUTOTUNE)\
            .shuffle(buffer_size=tf.data.AUTOTUNE,
                     seed=SEED)\
            .batch(EVAL_BATCH)\
            .repeat()\
            .cache()\
            .prefetch(tf.data.AUTOTUNE)

        test_dataset = test_dataset.map(
            lambda x, y: self._tokenizer_fn(x, y),
            num_parallel_calls=tf.data.AUTOTUNE)\
            .shuffle(buffer_size=tf.data.AUTOTUNE,
                     seed=SEED)\
            .batch(EVAL_BATCH)\
            .repeat()\
            .cache()\
            .prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

    def train(self, td_metadata_in: str,
              checkpoints: str,
              scores_out: str,
              state: str | None = None):
        """Performs model training and evaluation.

        Args:
            td_metadata_in (str): Training data metadata.
            state (str): The checkpoint path to use as state to start training.
            checkpoints (str): Checkpointing location.
        """
        logger.info("⏳ Starting Training task...🔄")
        # TODO: Import metrics as a module
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=AIP_TENSORBOARD,
            histogram_freq=TENSORBOARD_HFREQ,
            update_freq=TENSORBOARD_UFREQ)
        # We can consider only checkpoint after since it's always the best
        # model
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoints, "checkpoint.model.keras"),
            monitor='val_loss',
            mode='min',
            save_freq='epoch')

        train_dataset, val_dataset, test_dataset = self._prepare_dataset(
            td_metadata_in)
        # Virtual epochs: A design pattern useful as new data arrives
        # 1000 because len(ds) is a multiple of 1000
        # Adjust STOP_POINT in the config file to get the effective
        # size you want to train on.
        TOTAL_TRAINING_EXAMPLES = int(STOP_POINT * 1000)
        steps_per_epoch = (TOTAL_TRAINING_EXAMPLES //
                           (BATCH * NUM_CHECKPOINTS))
        model = self._import_model(state)
        
        logger.info("🧠 Training model Training...🔄")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            callbacks=[tensorboard_callback,
                       checkpoint_callback],
            epochs=NUM_CHECKPOINTS,
            batch_size=BATCH,
            steps_per_epoch=steps_per_epoch)
        eval_scores = model.evaluate(
            test_dataset,
            callbacks=[tensorboard_callback],
            return_dict=True)
        logger.info("🧠 Model training completed successfully!✅")

        if not os.path.exists(os.path.dirname(scores_out)):
            os.makedirs(os.path.dirname(scores_out))

        scores = {"history": history.history,
                  "eval_scores": eval_scores}

        logger.info("🛠️ Exporting metrics artefact...🔄")
        with open(scores_out, 'w') as file:
            file.write(json.dumps(scores))

        logger.info("⌛️ Training completed successfully!✅")


if __name__ == "__main__":
    Fire(Train)
