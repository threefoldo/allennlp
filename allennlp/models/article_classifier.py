
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


@Model.register('article_classifier')
class ArticleClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ArticleClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = vocab.get_vocab_size('labels')
        print('vocabulary:', vocab.get_index_to_token_vocabulary('labels'), '\n')
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != title_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            title_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != abstract_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)


    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> "ArticleClassifier":
        embedder_params = params.pop('text_field_embedder')
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        title_encoder = Seq2VecEncoder.from_params(params.pop('title_encoder'))
        abstract_encoder = Seq2VecEncoder.from_params(params.pop('abstract_encoder'))
        classifier_feedforward = FeedForward.from_params(params.pop('classifier_feedforward'))

        initializer = InitializerApplicator.from_params(params.pop('intializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab = vocab,
                   text_field_embedder = text_field_embedder,
                   title_encoder = title_encoder,
                   abstract_encoder = abstract_encoder,
                   classifier_feedforward = classifier_feedforward,
                   initializer = initializer,
                   regularizer = regularizer)


    @overrides
    def forward(self,
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        embedded_title = self.text_field_embedder(title)
        title_mask = get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_abstract = self.text_field_embedder(abstract)
        abstract_mask = get_text_field_mask(abstract)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract], dim=-1))
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


