import pytest

from app.config import get_settings
from app.errors import SampleNotFoundError
from app.facts import FactBook
from app.samples import SampleLibrary


def test_fact_book_loads_and_covers_demo_birds():
    facts = FactBook(get_settings().facts_file)
    assert facts.lookup("Turdus migratorius").habitat
    assert facts.lookup("Strix aluco").song
    assert facts.lookup("Unknown bird") is None


def test_sample_library_files_exist_with_credits():
    library = SampleLibrary(get_settings().samples_dir)
    ids = {s.id for s in library.list()}
    assert {"morning-robin", "deep-woods-owl", "windy-ridge"} <= ids
    for sample in library.list():
        assert library.path(sample.id).is_file()
        assert sample.credit and sample.license


def test_unknown_sample_raises():
    with pytest.raises(SampleNotFoundError):
        SampleLibrary(get_settings().samples_dir).get("nope")
