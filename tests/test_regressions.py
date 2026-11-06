import unittest
from unittest import mock

import embeddings


class EmbeddingTests(unittest.TestCase):
    def test_preprocessing_input_is_bounded(self):
        seen = []
        real = embeddings._preprocess_for_embedding

        def spy(text):
            seen.append(len(text))
            return real(text)

        model = mock.Mock()
        model.encode.return_value.tolist.return_value = [0.0]
        with mock.patch.object(embeddings, "_model", model), \
                mock.patch.object(embeddings, "_model_type", "sentence-transformers"), \
                mock.patch.object(embeddings, "_preprocess_for_embedding", spy):
            embeddings.generate_embedding("![" * 50000, max_chars=2000)
        self.assertEqual(seen, [2000 * embeddings._PREPROCESS_WINDOW])
