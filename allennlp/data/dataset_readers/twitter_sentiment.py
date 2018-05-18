

from typing import Dict
import logging
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer, CharacterTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)

@DatasetReader.register('twitter-sentiment')
class TwitterSentimentDatasetReader(DatasetReader):
    '''
    one sample per line: tweet_id user_id label text
    '''

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False ) -> None:
        super().__init__(lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @classmethod
    def from_params(cls, params: Params) -> 'Twitter1DatasetReader':
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
                line = line.split(',')
                if len(line) < 4:
                    continue
                sent_id = line[0].strip()
                label  = line[1].strip()
                text  = line[3].strip()
                if len(text) > 0 and label in ['0', '1']:
                    yield self.text_to_instance(text, label)


    @overrides
    def text_to_instance(self, text:str, label:str = None) -> Instance:
        text_field = TextField(self._tokenizer.tokenize(text), self._token_indexers)
        fields = {'text': text_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)



