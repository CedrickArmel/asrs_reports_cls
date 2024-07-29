"""Defines the components of the Train pipeline."""

import json
import os
from typing import Optional

import hopsworks
import keras
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fire import Fire
from hsfs.client.exceptions import RestAPIError
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.train.models import CustomModelForSequenceClassification as Cls
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
STOP_POINTS = trainconf.training.stop_points

project = hopsworks.login()
fs = project.get_feature_store()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class Train:
    """Train pipeline components"""

    def __init__(
        self,
        batch: int = BATCH,
        eval_batch: int = EVAL_BATCH,
        fg_name: str = FG_NAME,
        fg_version: int = FG_VERSION,
        fv_name: str = FV_NAME,
        fv_version: int = FV_VERSION,
        labels: list = LABELS,
        metrics: list = METRICS,
        model_name: list = MODEL_NAME,
        num_checkpoints: int = NUM_CHECKPOINTS,
        optimizer: str = OPTIMIZER,
        aip_tensorboard: Optional[str] = AIP_TENSORBOARD,
        tensorboard_hfreq: int = TENSORBOARD_HFREQ,
        tensorboard_ufreq: int = TENSORBOARD_UFREQ,
        td_description: str = TD_DESCRIPTION,
        td_format: str = TD_FORMAT,
        td_new: bool = TD_NEW,
        td_version: int = TD_VERSION,
        seed: int = SEED,
        stop_points: float = STOP_POINTS,
    ):
        self.batch = batch
        self.eval_batch = eval_batch
        self.fg_name = fg_name
        self.fg_version = fg_version
        self.fv_name = fv_name
        self.fv_version = fv_version
        self.labels = labels
        self.metrics = metrics
        self.model_name = model_name
        self.num_checkpoints = num_checkpoints
        self.optimizer = optimizer
        self.aip_tensorboard = aip_tensorboard
        self.tensorboard_hfreq = tensorboard_hfreq
        self.tensorboard_ufreq = tensorboard_ufreq
        self.td_description = td_description
        self.td_format = td_format
        self.td_new = td_new
        self.td_version = td_version
        self.seed = seed
        self.stop_points = stop_points

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.strategy = tf.distribute.MirroredStrategy()

    def create_training_data(
        self,
        td_metadata_output: str,
    ):
        """Create a training data in the Hopsworks Feature Store.

        Args:
            td_metada_output (str): Path where to output \
                the training data metadata artefact.
        """
        logger.info("⏳ Starting Create Training Data task...🔄")

        try:
            fv = fs.get_feature_view(
                name=self.fv_name,
                version=self.fv_version,
            )
        except RestAPIError as e:
            logger.warning(
                "⚠️ %s raised. Feature View seems missing. \
                           Trying to create a new one...🔄",
                str(e),
            )
            try:
                fg = fs.get_feature_group(
                    name=self.fg_name,
                    version=self.fg_version,
                )
                # TODO: Externalise query as a module
                query = fg.select_all(include_primary_key=False)
                fv = fs.create_feature_view(
                    name=self.fv_name,
                    version=self.fv_version,
                    query=query,
                )
            except RestAPIError as e2:
                logger.exception(
                    "💀 %s raised. Feature View creation failed ❌. \
                        Seems that Feature Group is also missing. \
                            A Feature Group is need before running \
                                this pipeline !",
                    str(e2),
                )

        (
            version,
            _,
        ) = fv.create_train_validation_test_split(
            validation_size=0.3,
            test_size=0.2,
            description=self.td_description,
            seed=self.seed,
            data_format=self.td_format,
            write_options={"use_spark": True},
        )
        metadata = json.loads(fv.json())

        if (version > self.td_version) and self.td_new:
            logger.warning(
                "⚠️ Using the training data version %s as verion %s \
                    already exists AND 'td_new' parameter have been \
                        set to 'True'. to change this behavior, \
                            set 'td_new' to 'false'.",
                str(version),
                str(self.td_version),
            )
            metadata["training_data_version"] = version
        else:
            metadata["training_data_version"] = self.td_version

        if not os.path.exists(os.path.dirname(td_metadata_output)):
            os.makedirs(os.path.dirname(td_metadata_output))

        with open(
            td_metadata_output,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(metadata))

        logger.info("⌛️ Create Training Data task completed successfully!✅")

    def _import_training_data(
        self,
        td_metadata_in: str,
    ) -> tuple:
        """Import the training data from Hopsworks Feature Store.

        Args:
            td_metadata (str): Path of the training data metadata.

        Returns:
            Tuple[pd.DataFrame]: (Features Dataframe, Target DataFrame).
        """
        with open(
            td_metadata_in,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        fv_name = metadata["name"]
        fv_version = metadata["version"]
        td_version = metadata["training_data_version"]
        fv = fs.get_feature_view(
            name=fv_name,
            version=fv_version,
        )
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = fv.get_train_validation_test_split(
            training_dataset_version=td_version
        )
        train_data = (
            X_train,
            y_train,
        )
        val_data = (
            X_val,
            y_val,
        )
        test_data = (
            X_test,
            y_test,
        )
        return (
            train_data,
            val_data,
            test_data,
        )

    @tf.py_function(
        Tout=[
            tf.int32,
            tf.int16,
        ]
    )
    def _tokenizer_fn(
        self,
        x,
        y,
    ):
        x = tf.compat.as_str(x.numpy()[0])
        x = self.tokenizer(
            x,
            None,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="tf",
        )
        x = tf.convert_to_tensor([v for v in x.values()])
        y = tf.convert_to_tensor(
            np.array(
                y,
                dtype=np.int16,
            )
        )
        return (
            x,
            y,
        )

    def _import_model(
        self,
        state: str | None = None,
    ) -> Cls:
        loss = keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0.0,
            axis=-1,
            reduction="sum_over_batch_size",
            name="binary_crossentropy",
        )
        if state != "no_checkpoint":
            logger.info("🏁 Importing model from checkpoint...🔄")
            with self.strategy.scope():
                model = keras.models.load_model(state)
        else:
            logger.info("🛠️ Initialising model from HuggingFace🤗...🔄")
            with self.strategy.scope():
                model = Cls.from_pretrained(
                    self.model_name,
                    num_labels=len(self.labels),
                    from_pt=True,
                )
                model.compile(
                    optimizer=self.optimizer,
                    loss=loss,
                    metrics=self.metrics,
                )
        logger.info("⌛️ Model import completed successfully!✅")
        return model

    def _prepare_dataset(
        self,
        td_metadata_in: str,
    ):
        (
            train_data,
            val_data,
            test_data,
        ) = self._import_training_data(td_metadata_in=td_metadata_in)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [item for item in train_data[0].values.tolist()],
                [item for item in train_data[1].values.tolist()],
            )
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [item for item in val_data[0].values.tolist()],
                [item for item in val_data[1].values.tolist()],
            )
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                [item for item in test_data[0].values.tolist()],
                [item for item in test_data[1].values.tolist()],
            )
        )

        # Below we repeat infinitely because we want to implement virtual
        # epochs and contant training sample size (as new data arrives)
        train_dataset = (
            train_dataset.map(
                lambda x, y: self._tokenizer_fn(
                    x,
                    y,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(
                buffer_size=tf.data.AUTOTUNE,
                seed=self.seed,
            )
            .batch(self.batch)
            .repeat()
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            val_dataset.map(
                lambda x, y: self._tokenizer_fn(
                    x,
                    y,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(
                buffer_size=tf.data.AUTOTUNE,
                seed=self.seed,
            )
            .batch(self.eval_batch)
            .repeat()
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        test_dataset = (
            test_dataset.map(
                lambda x, y: self._tokenizer_fn(
                    x,
                    y,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .shuffle(
                buffer_size=tf.data.AUTOTUNE,
                seed=self.seed,
            )
            .batch(self.eval_batch)
            .repeat()
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
        )

    def train(
        self,
        td_metadata_in: str,
        checkpoints: str,
        scores_out: str,
        state: str | None = None,
    ):
        """Performs model training and evaluation.

        Args:
            td_metadata_in (str): Training data metadata.
            state (str): The checkpoint path to use as state to start training.
            checkpoints (str): Checkpointing location.
        """
        logger.info("⏳ Starting Training task...🔄")
        # TODO: Import metrics as a module
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.aip_tensorboard,
            histogram_freq=self.tensorboard_hfreq,
            update_freq=self.tensorboard_ufreq,
        )
        # We can consider only checkpoint after since it's always the best
        # model
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoints,
                "checkpoint.model.keras",
            ),
            monitor="val_loss",
            mode="min",
            save_freq="epoch",
        )

        (
            train_dataset,
            val_dataset,
            test_dataset,
        ) = self._prepare_dataset(td_metadata_in)
        # Virtual epochs: A design pattern useful as new data arrives
        # 1000 because len(ds) is a multiple of 1000
        # Adjust STOP_POINT in the config file to get the effective
        # size you want to train on.
        total_training_examples = int(self.stop_points * 1000)
        steps_per_epoch = total_training_examples // (
            self.batch * self.num_checkpoints
        )
        model = self._import_model(state)

        logger.info("🧠 Training model Training...🔄")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            callbacks=[
                tensorboard_callback,
                checkpoint_callback,
            ],
            epochs=self.num_checkpoints,
            batch_size=self.batch,
            steps_per_epoch=steps_per_epoch,
        )
        eval_scores = model.evaluate(
            test_dataset,
            callbacks=[tensorboard_callback],
            return_dict=True,
        )
        logger.info("🧠 Model training completed successfully!✅")

        if not os.path.exists(os.path.dirname(scores_out)):
            os.makedirs(os.path.dirname(scores_out))

        scores = {
            "history": history.history,
            "eval_scores": eval_scores,
        }

        logger.info("🛠️ Exporting metrics artefact...🔄")
        with open(
            scores_out,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(scores))

        logger.info("⌛️ Training completed successfully!✅")


if __name__ == "__main__":
    Fire(Train)
