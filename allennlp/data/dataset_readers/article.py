

from typing import Dict
import logging
import json
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)

@DatasetReader.register('article')
class ArticleDatasetReader(DatasetReader):
    '''
    input format: one JSON per line, {title, abstract, venue}
    '''

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False ) -> None:
        super().__init__(lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @classmethod
    def from_params(cls, params: Params) -> 'ArticleDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        return cls(tokenizer = tokenizer, token_indexers = token_indexers, lazy=lazy)


    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path)) as fp:
            logger.info('Reading file %s' % file_path)
            for line in fp.readlines():
                line = line.strip()
                if line is None:
                    continue
                data = json.loads(line)
                title = data['title']
                abstract  = data['abstract']
                venue = data['venue']
                yield self.text_to_instance(title, abstract, venue)


    @overrides
    def text_to_instance(self, title:str, abstract:str, venue:str = None) -> Instance:
        title_field = TextField(self._tokenizer.tokenize(title), self._token_indexers)
        abstract_field = TextField(self._tokenizer.tokenize(abstract), self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)



