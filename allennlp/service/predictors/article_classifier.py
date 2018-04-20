from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('article_predictor')
class ArticleClassifierPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.article_classifier` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        title = json_dict.get('title') or json_dict.get('word')
        abstract = json_dict.get('abstract') or 'none'
        instance = self._dataset_reader.text_to_instance(title, abstract)

        # label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        # all_labels = [label_dict[i] for i in range(len(label_dict))]
        # return instance, {"all_labels": all_labels}
        return instance, {}

