import pytest

from allennlp.data.dataset_readers import ArticleDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestArticleDatasetReader(AllenNlpTestCase):

    def test_read_from_file(self):
        reader = ArticleDatasetReader()
        dataset = reader.read('tests/fixtures/articles.jsonl')

        instance1 = {
            "title": ["张", "铭", "爽"],
            "abstract": ["n", "o", "n", "e"],
            "label": "人名"
        }
        instance2 = {
            "title": ["白", "先", "勇"],
            "abstract": ["n", "o", "n", "e"],
            "label": "人名"
        }

        assert len(dataset) == 10

        fields = dataset[0].fields
        assert [t.text for t in fields['title'].tokens] == instance1['title']
        assert [t.text for t in fields['abstract'].tokens[:5]] == instance1['abstract']
        assert fields['label'].label == instance1['label']

