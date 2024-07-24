"""This module contains models defintion for the Train pipeline."""
from typing import Optional, Tuple
import tensorflow as tf
from transformers.models.bert.modeling_tf_bert import (
    TFBertForSequenceClassification)
from transformers import BertConfig


class CustomModelForSequenceClassification(TFBertForSequenceClassification):
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._set_layers_to_update()
        self.build()

    def _set_layers_to_update(self):
        self.bert.trainable = False
        self.bert.pooler.trainable = True
        self.classifier.trainable = True
        for layer in self.bert.encoder.layer[8:]:
            layer.trainable = False

    def call(self, x: tf.Tensor,
             training: Optional[bool] = True):
        if x.shape.rank == 3:
            input_ids = x[:, 0, :]
            token_type_ids = x[:, 1, :]
            attention_mask = x[:, 2, :]
        elif x.shape.rank == 2:
            input_ids = tf.reshape(x[0], (1, -1))
            token_type_ids = tf.reshape(x[1], (1, -1))
            attention_mask = tf.reshape(x[2], (1, -1))
        else:
            raise ValueError("Invalid input shape")
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=training)
        logits = self.classifier(inputs=pooled_output)
        return logits

    def train_step(self,
                   data: Tuple[tf.Tensor, tf.Tensor]) -> dict:
        """Performs the training step when fit() is called.

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): (x,y) like item\
                returned by a tf.data.Dataset.

        Returns:
            dict: The computed metrics passed to compile().
        """
        x, y = data
        x = tf.ensure_shape(x, [None, 3, 512])
        y = tf.reshape(y, (-1, self.num_labels)) 
        y = tf.ensure_shape(y, [None, 14])
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self,
                  data: Tuple[tf.Tensor, tf.Tensor]) -> dict:
        """Performs the evaluation step when evaluate() is called.

        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): (x,y) like item\
                returned by a tf.data.Dataset.

        Returns:
            dict: The computed metrics passed to compile().
        """
        x, y = data
        x = tf.ensure_shape(x, [None, 3, 512])
        y = tf.reshape(y, (-1, self.num_labels))
        y = tf.ensure_shape(y, [None, 14])
        y_pred = self(x, training=False)
        self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
