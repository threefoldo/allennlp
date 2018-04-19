import json
import logging
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer
from allennlp.data.fields import TextField, LabelField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("jsonl")
class JsonlReader(DatasetReader):
    """
    Reads a JSON-formatted file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or CharacterTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                if line is None or line.strip() == '':
                    continue
                dataset.append(json.loads(line))

        logger.info("Reading the dataset")
        for data in dataset:
            yield self.text_to_instance(data['word'], data['label'])


    @overrides
    def text_to_instance(self, word:str, label:str) -> Instance:
        f_word = TextField(self._tokenizer.tokenize(word), self._token_indexers)
        f_label = LabelField(label)
        return Instance({'word': f_word, 'label': f_label})


    @classmethod
    def from_params(cls, params: Params) -> 'JsonlReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)
