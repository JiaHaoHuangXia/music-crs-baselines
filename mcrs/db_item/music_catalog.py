import os
import json
from datasets import load_dataset, concatenate_datasets
import torch
from mcrs.controlled_tags import metadata_field
from mcrs.style_profiles import attach_artist_style_profiles

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        self.metadata_dict = attach_artist_style_profiles(self.metadata_dict)

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        metadata = self.metadata_dict[track_id]
        track_id = metadata['track_id']
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            value = metadata_field(metadata, corpus_type)
            if isinstance(value, list):
                corpus_type_value = ", ".join(str(item) for item in value).lower()
            else:
                corpus_type_value = str(value).lower()
            entity_str += f", {corpus_type}: {corpus_type_value}"
        return entity_str
