"""Offline tests for the per-instance annotate() cache."""

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation,
)

PROV = AnnotationProvenance("Counting", "http://counting.example/", "1")


class CountingLinker(Annotator):
    def __init__(self):
        self.calls = []

    def _annotate(self, text, props):
        self.calls.append((text, props))
        return AnnotatedText(
            text,
            [
                NormalizedAnnotation(
                    text=text,
                    id=f"X:{len(self.calls)}",
                    label=text,
                    type="biolink:NamedThing",
                    biolink_type="biolink:NamedThing",
                    start=0,
                    end=len(text),
                    provenance=PROV,
                )
            ],
        )


def test_cache_keys_on_text_and_props():
    linker = CountingLinker()
    first = linker.annotate("brain", {"limit": 1})
    assert linker.annotate("brain", {"limit": 1}) is first
    linker.annotate("brain", {"limit": 2})
    linker.annotate("heart", {"limit": 1})
    assert linker.calls == [
        ("brain", {"limit": 1}),
        ("brain", {"limit": 2}),
        ("heart", {"limit": 1}),
    ]
    # skip_cache bypasses the cache and is not passed on to the service.
    linker.annotate("brain", {"limit": 1, "skip_cache": True})
    assert linker.calls[-1] == ("brain", {"limit": 1})
    assert len(linker.calls) == 4


def test_cache_can_be_disabled():
    linker = CountingLinker()
    linker.cache_size = 0
    linker.annotate("brain")
    linker.annotate("brain")
    assert len(linker.calls) == 2


def test_reannotate_does_not_corrupt_cached_results():
    """reannotate() shifts offsets; doing that in place would break the second use of a cached result."""
    ner = AnnotationProvenance("NER", "http://ner.example/", "1")
    linker = CountingLinker()
    for text, start in [("The brain.", 4), ("A brain!", 2)]:
        source = AnnotatedText(
            text,
            [
                Annotation(
                    "brain", "I1", "", "biolink:NamedThing", start, start + 5, ner
                )
            ],
        )
        (ann,) = source.reannotate(linker).annotations
        assert (ann.start, ann.end) == (start, start + 5)
        assert [p.name for p in ann.provenances] == ["NER", "Counting"]
    assert len(linker.calls) == 1
    # The cached object itself is untouched.
    cached = linker.annotate("brain").annotations[0]
    assert (cached.start, cached.end, cached.based_on) == (0, 5, [])


def test_cache_key_accepts_list_valued_props():
    """NameRes takes lists (biolink_types, only_prefixes); they must be a valid cache key."""
    linker = CountingLinker()
    linker.annotate(
        "brain", {"biolink_types": ["biolink:AnatomicalEntity"], "limit": 1}
    )
    linker.annotate(
        "brain", {"limit": 1, "biolink_types": ["biolink:AnatomicalEntity"]}
    )
    linker.annotate("brain", {"biolink_types": ["biolink:Gene"], "limit": 1})
    assert len(linker.calls) == 2


def test_pipeline_cache_hit_does_not_rerun_steps():
    from renci_ner.core import Pipeline

    class CountingNER(Annotator):
        def __init__(self):
            self.calls = 0

        def _annotate(self, text, props):
            self.calls += 1
            return AnnotatedText(
                text, [Annotation("brain", "I1", "", "biolink:NamedThing", 4, 9, PROV)]
            )

    ner, linker = CountingNER(), CountingLinker()
    pipeline = Pipeline(ner, linker)
    first = pipeline.annotate("The brain.")
    assert pipeline.annotate("The brain.") is first
    assert (ner.calls, len(linker.calls)) == (1, 1)
