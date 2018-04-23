from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('simple_seq2seq')
class SimpleSeq2SeqPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.encoder_decoder.simple_seq2seq` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict.get("source") or json_dict.get('sentence')
        return self._dataset_reader.text_to_instance(source), {}
