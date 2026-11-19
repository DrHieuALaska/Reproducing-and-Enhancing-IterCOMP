"""Monotonic stable-ID union, rendered in original document and sentence order."""


class EvidenceAccumulator:
    def __init__(self, segments):
        self.catalog = {s.id: s for s in segments}
        if len(self.catalog) != len(segments):
            raise ValueError("Duplicate catalog IDs")
        self.selected = set()

    def update(self, retrieved_ids):
        if not set(retrieved_ids) <= self.catalog.keys():
            raise ValueError("Unknown evidence ID")
        self.selected.update(retrieved_ids)
        return self.ids()

    def ids(self):
        return [sid for sid in self.catalog if sid in self.selected]

    def segments(self):
        return [self.catalog[sid] for sid in self.ids()]
