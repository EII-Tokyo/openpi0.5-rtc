from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import tyro

from examples.droid.convert_legacy_lerobot_to_canonical import LegacyConversionConfig
from examples.droid.convert_legacy_lerobot_to_canonical import main as convert_legacy_main
from examples.droid.convert_raw_droid_to_canonical_lerobot import RawConversionConfig
from examples.droid.convert_raw_droid_to_canonical_lerobot import main as convert_raw_main


@dataclass
class RebuildCanonicalDroidConfig:
    destination_repo_id: str
    destination_root: Path | None = None
    mongo_url: str | None = None
    mongo_db_name: str = "eii_data_system"
    mongo_project_path_filters: list[str] | None = None
    mongo_project_date_filters: list[str] | None = None
    annotations_path: Path | None = None
    legacy_source_repo_ids: list[str] | None = None
    legacy_source_roots: list[Path] | None = None
    legacy_raw_data_roots: list[Path] | None = None
    raw_data_dir: Path | None = None
    push_to_hub: bool = False
    overwrite: bool = False
    suppress_encoder_output: bool = True
    batch_encoding_size: int = 1
    image_writer_processes: int = 0
    image_writer_threads: int = 8


def main(config: RebuildCanonicalDroidConfig) -> None:
    ran_legacy_phase = False
    if config.legacy_source_repo_ids:
        ran_legacy_phase = True
        convert_legacy_main(
            LegacyConversionConfig(
                source_repo_ids=config.legacy_source_repo_ids,
                destination_repo_id=config.destination_repo_id,
                destination_root=config.destination_root,
                source_roots=config.legacy_source_roots,
                raw_data_roots=config.legacy_raw_data_roots,
                annotations_path=config.annotations_path,
                overwrite=config.overwrite,
                resume=False,
                push_to_hub=False,
                batch_encoding_size=config.batch_encoding_size,
                image_writer_processes=config.image_writer_processes,
                image_writer_threads=config.image_writer_threads,
                suppress_encoder_output=config.suppress_encoder_output,
                mongo_url=config.mongo_url,
                mongo_db_name=config.mongo_db_name,
                mongo_project_path_filters=config.mongo_project_path_filters,
                mongo_project_date_filters=config.mongo_project_date_filters,
            )
        )

    if config.raw_data_dir is not None:
        convert_raw_main(
            RawConversionConfig(
                data_dir=config.raw_data_dir,
                repo_id=config.destination_repo_id,
                output_root=config.destination_root,
                annotations_path=config.annotations_path,
                push_to_hub=config.push_to_hub,
                overwrite=config.overwrite and not ran_legacy_phase,
                resume=ran_legacy_phase,
                batch_encoding_size=config.batch_encoding_size,
                image_writer_processes=config.image_writer_processes,
                image_writer_threads=config.image_writer_threads,
                suppress_encoder_output=config.suppress_encoder_output,
                mongo_url=config.mongo_url,
                mongo_db_name=config.mongo_db_name,
                mongo_project_path_filters=config.mongo_project_path_filters,
                mongo_project_date_filters=config.mongo_project_date_filters,
            )
        )
        return

    if ran_legacy_phase and config.push_to_hub:
        logging.info("Legacy-only rebuild completed locally; push separately after review if desired.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(tyro.cli(RebuildCanonicalDroidConfig))
