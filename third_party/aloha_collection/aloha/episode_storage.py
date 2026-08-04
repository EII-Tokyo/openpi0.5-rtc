"""Exclusive staging and atomic publication for recorded episodes."""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class EpisodeClaimCollision(FileExistsError):
    """Raised when an episode index already has a lexical claim entry."""


class EpisodePublishCollision(FileExistsError):
    """Raised when an episode destination is occupied at publication time."""


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def _dataset_lock(dataset_dir: Path):
    lock_path = dataset_dir / ".aloha-recording.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


@dataclass
class EpisodeClaim:
    """An exclusive, token-identified claim on one episode index."""

    dataset_dir: Path
    index: int
    claim_path: Path
    owner_token: str
    _claim_identity: tuple[int, int]
    _released: bool = False

    @classmethod
    def acquire(
        cls,
        dataset_dir: str | os.PathLike[str],
        index: int,
    ) -> "EpisodeClaim":
        dataset_path = Path(dataset_dir)
        claim_path = dataset_path / f".episode_{index}.claim"
        owner_token = str(uuid.uuid4())
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        with _dataset_lock(dataset_path):
            try:
                claim_fd = os.open(claim_path, flags, 0o600)
            except FileExistsError as exc:
                raise EpisodeClaimCollision(
                    f"episode {index} is already claimed"
                ) from exc

            try:
                os.write(claim_fd, owner_token.encode("utf-8"))
                claim_stat = os.fstat(claim_fd)
            except BaseException:
                os.close(claim_fd)
                try:
                    if claim_path.read_text(encoding="utf-8") == owner_token:
                        claim_path.unlink()
                except (FileNotFoundError, IsADirectoryError, OSError):
                    pass
                raise
            else:
                os.close(claim_fd)

        return cls(
            dataset_dir=dataset_path,
            index=index,
            claim_path=claim_path,
            owner_token=owner_token,
            _claim_identity=(claim_stat.st_dev, claim_stat.st_ino),
        )

    def release(self) -> bool:
        """Remove this claim only while its exact owner token is present."""
        if self._released:
            return True
        with _dataset_lock(self.dataset_dir):
            try:
                token = self.claim_path.read_text(encoding="utf-8")
                claim_stat = self.claim_path.lstat()
            except (FileNotFoundError, IsADirectoryError, OSError):
                return False
            if (
                token != self.owner_token
                or (claim_stat.st_dev, claim_stat.st_ino)
                != self._claim_identity
            ):
                return False
            try:
                self.claim_path.unlink()
            except (FileNotFoundError, IsADirectoryError, OSError):
                return False
            self._released = True
            return True

    def __enter__(self) -> "EpisodeClaim":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


@dataclass
class StagedEpisode:
    """An owned staging directory coupled to an exclusive episode claim."""

    dataset_dir: Path
    index: int
    claim: EpisodeClaim
    staging_path: Path
    _staging_identity: tuple[int, int]
    _state: str = "staged"

    @classmethod
    def create(
        cls,
        dataset_dir: str | os.PathLike[str],
        index: int,
    ) -> "StagedEpisode":
        dataset_path = Path(dataset_dir)
        claim = EpisodeClaim.acquire(dataset_path, index)
        try:
            staging_path = Path(
                tempfile.mkdtemp(
                    prefix=f".episode_{index}.staging-",
                    dir=dataset_path,
                )
            )
            stat_result = staging_path.stat()
        except BaseException:
            claim.release()
            raise
        return cls(
            dataset_dir=dataset_path,
            index=index,
            claim=claim,
            staging_path=staging_path,
            _staging_identity=(stat_result.st_dev, stat_result.st_ino),
        )

    def _owns_staging_directory(self) -> bool:
        try:
            stat_result = self.staging_path.lstat()
        except FileNotFoundError:
            return False
        return (
            self.staging_path.is_dir()
            and not self.staging_path.is_symlink()
            and (stat_result.st_dev, stat_result.st_ino)
            == self._staging_identity
        )

    def discard(self) -> None:
        """Discard this object's staging directory without touching a final."""
        if self._state == "published":
            return
        if self._state == "discarded":
            self.claim.release()
            return
        if self._owns_staging_directory():
            shutil.rmtree(self.staging_path)
        self.claim.release()
        self._state = "discarded"

    def publish(
        self,
        *,
        allow_existing_destination: bool = False,
        data_suffix: str = "hdf5",
    ) -> Path:
        """Atomically rename this staging directory to its final destination."""
        if self._state != "staged":
            raise RuntimeError(f"cannot publish episode in {self._state} state")
        if not self._owns_staging_directory():
            raise RuntimeError("owned episode staging directory is unavailable")

        final_path = self.dataset_dir / f"episode_{self.index}"
        legacy_path = self.dataset_dir / f"episode_{self.index}.{data_suffix}"
        lock_path = self.dataset_dir / ".aloha-recording.lock"
        lock_flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            lock_flags |= os.O_CLOEXEC
        lock_fd = os.open(lock_path, lock_flags, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if _path_entry_exists(legacy_path):
                raise EpisodePublishCollision(
                    f"legacy episode destination exists: {legacy_path}"
                )
            final_exists = _path_entry_exists(final_path)
            if final_exists:
                if (
                    not allow_existing_destination
                    or final_path.is_symlink()
                    or not final_path.is_dir()
                ):
                    raise EpisodePublishCollision(
                        f"episode destination exists: {final_path}"
                    )
            os.chmod(self.staging_path, 0o755)
            if final_exists:
                self._replace_existing_directory(final_path)
            else:
                os.rename(self.staging_path, final_path)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

        self._state = "published"
        self.claim.release()
        return final_path

    def _replace_existing_directory(self, final_path: Path) -> None:
        backup_path = self._unique_backup_path()
        os.rename(final_path, backup_path)
        staging_published = False
        try:
            os.rename(self.staging_path, final_path)
            staging_published = True
            shutil.rmtree(backup_path)
        except BaseException:
            if (
                staging_published
                and _path_entry_exists(final_path)
                and not _path_entry_exists(self.staging_path)
            ):
                os.rename(final_path, self.staging_path)
            if _path_entry_exists(backup_path) and not _path_entry_exists(
                final_path
            ):
                os.rename(backup_path, final_path)
            raise

    def _unique_backup_path(self) -> Path:
        while True:
            backup_path = (
                self.dataset_dir
                / f".episode_{self.index}.backup-{uuid.uuid4().hex}"
            )
            if not _path_entry_exists(backup_path):
                return backup_path
