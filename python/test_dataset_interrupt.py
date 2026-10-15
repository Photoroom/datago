"""The iterator adapter must not convert Ctrl-C into normal end-of-data."""

import dataset
import pytest


def test_keyboard_interrupt_from_client_is_reraised_after_stop(monkeypatch):
    class Client:
        def __init__(self, config):
            self.stopped = 0

        def start(self):
            pass

        def get_sample(self):
            raise KeyboardInterrupt

        def get_sample_auto_convert(self):
            raise KeyboardInterrupt

        def stop(self):
            self.stopped += 1

    monkeypatch.setattr(dataset, "DatagoClient", Client)
    iterator = dataset.DatagoIterDataset({"limit": 1}, return_python_types=False)
    with pytest.raises(KeyboardInterrupt):
        next(iterator)
    assert iterator.client.stopped == 1
