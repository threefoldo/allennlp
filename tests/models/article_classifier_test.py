import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params


class ArticleClassifierTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/article_classifier.json',
                          'tests/fixtures/articles.jsonl')

    def test_train_load_save(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
