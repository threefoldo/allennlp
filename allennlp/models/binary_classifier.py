
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from typing import Dict, Optional
import torch
import numpy as np
from overrides import overrides
import torch.nn.functional as F


@Model.register('binary_classifier')
class BinaryClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BinaryClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size('labels')

        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            encoder.get_input_dim()))

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)


    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "BinaryClassifier":
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2VecEncoder.from_params(params.pop('encoder'))
        classifier_feedforward = FeedForward.from_params(params.pop('classifier_feedforward'))

        initializer = InitializerApplicator.from_params(params.pop('intializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab = vocab,
                   text_field_embedder = text_field_embedder,
                   encoder = encoder,
                   classifier_feedforward = classifier_feedforward,
                   initializer = initializer,
                   regularizer = regularizer)


    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_text = self.text_field_embedder(text)
        text_mask = get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, text_mask)

        logits = self.classifier_feedforward(encoded_text)
        # class_probabilities = F.softmax(logits, dim=0)

        # output_dict = {'class_probabilities': class_probabilities}
        output_dict = {'logits': logits}
        if label is not None:
            # print(logits, label)
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict['loss'] = loss

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict


