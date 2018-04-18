import pytest

from allennlp.data.dataset_readers import ArticleDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestArticleDatasetReader(AllenNlpTestCase):

    def test_read_from_file(self):
        reader = ArticleDatasetReader()
        dataset = reader.read('tests/fixtures/articles.jsonl')

        instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}

        instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                     "abstract": ["Execut", "ion", "of", "classical", "plans"],
                     "venue": "AI"}
        instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
                               "Traveller", "Problem"],
                     "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                     "venue": "AI"}

        #assert len(dataset.instances) == 10

        fields = dataset[0].fields
        assert [t.text for t in fields['title'].tokens] == instance1['title']
        assert [t.text for t in fields['abstract'].tokens[:5]] == instance1['abstract']
        assert fields['label'].label == instance1['venue']

