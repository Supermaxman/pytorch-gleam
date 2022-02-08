
import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds.multi_class import MultiClassThresholdModule
from pytorch_gleam.modeling.metrics.multi_class_f1 import F1PRMultiClassMetric


# noinspection PyAbstractClass
class MultiClassLanguageModel(BaseLanguageModel):
	def __init__(
			self,
			num_classes: int = 3,
			metric: str = 'f1',
			metric_mode: str = 'macro',
			*args,
			**kwargs
	):
		r"""
		Multi-Class Language Model for baseline n-way classification tasks.
			.. deprecated:: v0.5.0
				Please use :class:`~pytorch_gleam.modeling.models.MultiClassFrameLanguageModel`
		Args:

			num_classes: Number of different classes, such as "Accept", "Reject", and "No Stance".
				Default: ``3``.

			metric_mode: "macro" or "micro" f1 score to use for evaluation.
				Default: ``macro``.

			metric: Metric to use for evaluation. TODO currently only ``f1`` is available.
				Default: ``f1``.

			pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

			pre_model_type: Type of pre-trained model.
				Default: [`AutoModel`].

			learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
				``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly over the remaining
				``1.0-lr_warm_up`` training steps.

			weight_decay: How much weight decay to apply in the AdamW optimizer.
				Default: ``0.0``.

			lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
				Default: ``0.1``.

			load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be initialized.
				Cuts down on model load time if you plan on loading your model from a checkpoint, as there is no reason to
				initialize your model twice.
				Default: ``True``.

			torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
				Default: ``None``.

		"""
		super().__init__(*args, **kwargs)
		self.num_classes = num_classes
		self.cls_layer = torch.nn.Linear(
			in_features=self.hidden_size,
			out_features=self.num_classes
		)
		self.criterion = torch.nn.CrossEntropyLoss(
			reduction='none'
		)
		self.score_func = torch.nn.Softmax(
			dim=-1
		)
		self.threshold = MultiClassThresholdModule()
		# TODO select based on metric
		self.metric = F1PRMultiClassMetric(
			num_classes=self.num_classes,
			mode=metric_mode
		)

	def forward(self, batch):
		contextualized_embeddings = self.lm_step(
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			token_type_ids=batch['token_type_ids'],
		)
		cls_embedding = contextualized_embeddings[:, 0]
		logits = self.cls_layer(cls_embedding)
		return logits

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		batch_logits = self(batch)
		batch_scores = self.score_func(batch_logits)
		batch_preds = self.threshold(batch_scores)
		batch_ids = batch['ids']
		results = {
			'ids': batch_ids,
			'logits': batch_logits,
			'scores': batch_scores,
			'preds': batch_preds
		}
		return results

	def eval_epoch_end(self, outputs, stage):
		loss = torch.cat([x['loss'] for x in outputs], dim=0).mean()
		scores = torch.cat([x['scores'] for x in outputs], dim=0)
		labels = torch.cat([x['labels'] for x in outputs], dim=0)
		scores = scores.cpu()
		labels = labels.cpu()
		self.threshold.cpu()

		if stage == 'val':
			# select max f1 threshold
			max_threshold, max_metrics = self.metric.best(
				labels,
				scores,
				self.threshold
			)
			self.threshold.update_thresholds(max_threshold)
		preds = self.threshold(scores)

		f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(
			labels,
			preds
		)
		self.log(f'{stage}_loss', loss)
		self.log(f'{stage}_f1', f1)
		self.log(f'{stage}_p', p)
		self.log(f'{stage}_r', r)
		for t_idx, threshold in enumerate(self.threshold.thresholds):
			self.log(f'{stage}_threshold_{t_idx}', threshold)
		for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
			self.log(f'{stage}_{cls_index}_f1', c_f1)
			self.log(f'{stage}_{cls_index}_p', c_p)
			self.log(f'{stage}_{cls_index}_r', c_r)

		self.threshold.to(self.device)

	def training_step(self, batch, batch_idx):
		batch_logits = self(batch)
		batch_labels = batch['labels']
		batch_loss = self.criterion(
			batch_logits,
			batch_labels
		)
		loss = batch_loss.mean()
		self.log('train_loss', loss)
		result = {
			'loss': loss
		}
		return result

	def eval_step(self, batch, batch_idx, dataloader_idx=None):
		results = self.predict_step(batch, batch_idx, dataloader_idx)
		loss = self.criterion(
			results['logits'],
			batch['labels']
		)
		results['loss'] = loss
		results['labels'] = batch['labels']
		return results
