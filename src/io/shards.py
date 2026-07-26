"""
WebDataset-style tar shard I/O for OpenCompMech.

Avoids filesystem inode exhaustion by batching samples into tar archives.
Each shard contains ~1000 samples with consistent naming.
"""

import io
import tarfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, BinaryIO
from dataclasses import dataclass
import threading


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return None  # Skip arrays
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items() 
                if convert_numpy_types(v) is not None}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj if convert_numpy_types(x) is not None]
    else:
        return obj


@dataclass
class Sample:
    """A single dataset sample."""
    sample_id: str                    # e.g., "000042"
    density: np.ndarray               # (H, W) design density
    displacement: np.ndarray          # (2, H, W) displacement field
    stress_vm: np.ndarray             # (H, W) Von Mises stress
    input_tensor: Optional[np.ndarray] = None  # (8, H, W) input channels
    metadata: Optional[Dict[str, Any]] = None


class TarShardWriter:
    """
    Accumulates samples and writes to tar shards.
    
    Usage:
        writer = TarShardWriter("data/shards", "stiff", samples_per_shard=1000)
        for sample in generate_samples():
            writer.add_sample(sample)
        writer.close()  # Flush remaining samples
    """
    
    def __init__(
        self,
        output_dir: str,
        prefix: str = "shard",
        samples_per_shard: int = 1000,
        compression: str = ""  # "" for none, "gz" for gzip
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.samples_per_shard = samples_per_shard
        self.compression = compression
        
        self.buffer: List[Sample] = []
        self.shard_id = 0
        self.total_samples = 0
        self._lock = threading.Lock()
    
    def add_sample(self, sample: Sample) -> None:
        """Add a sample to the buffer. Flushes when buffer is full."""
        with self._lock:
            self.buffer.append(sample)
            if len(self.buffer) >= self.samples_per_shard:
                self._flush_shard()
    
    def _flush_shard(self) -> None:
        """Write current buffer to a tar shard."""
        if not self.buffer:
            return
        
        # Shard filename
        ext = f".tar.{self.compression}" if self.compression else ".tar"
        shard_path = self.output_dir / f"{self.prefix}_{self.shard_id:05d}{ext}"
        
        mode = f"w:{self.compression}" if self.compression else "w"
        
        with tarfile.open(shard_path, mode) as tar:
            for sample in self.buffer:
                self._add_sample_to_tar(tar, sample)
        
        print(f"Wrote shard {shard_path.name} ({len(self.buffer)} samples)")
        
        self.total_samples += len(self.buffer)
        self.buffer.clear()
        self.shard_id += 1
    
    def _add_sample_to_tar(self, tar: tarfile.TarFile, sample: Sample) -> None:
        """Add a single sample to the tar archive."""
        base_name = sample.sample_id
        
        # Density
        self._add_npy_to_tar(tar, f"{base_name}.density.npy", sample.density)
        
        # Displacement field
        self._add_npy_to_tar(tar, f"{base_name}.displacement.npy", sample.displacement)
        
        # Stress field
        self._add_npy_to_tar(tar, f"{base_name}.stress.npy", sample.stress_vm)
        
        # Input tensor (if provided)
        if sample.input_tensor is not None:
            self._add_npy_to_tar(tar, f"{base_name}.input.npy", sample.input_tensor)
        
        # Metadata
        if sample.metadata is not None:
            self._add_json_to_tar(tar, f"{base_name}.json", sample.metadata)
    
    def _add_npy_to_tar(
        self, tar: tarfile.TarFile, name: str, array: np.ndarray
    ) -> None:
        """Add numpy array to tar as .npy file."""
        buf = io.BytesIO()
        np.save(buf, array.astype(np.float32))
        buf.seek(0)
        
        info = tarfile.TarInfo(name=name)
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    
    def _add_json_to_tar(
        self, tar: tarfile.TarFile, name: str, data: Dict[str, Any]
    ) -> None:
        """Add JSON metadata to tar."""
        # Convert numpy types to Python types
        clean_data = convert_numpy_types(data)
        content = json.dumps(clean_data, indent=2).encode('utf-8')
        buf = io.BytesIO(content)
        
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tar.addfile(info, buf)
    
    def close(self) -> None:
        """Flush any remaining samples."""
        with self._lock:
            if self.buffer:
                self._flush_shard()
        print(f"Total samples written: {self.total_samples}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class TarShardReader:
    """
    Read samples from tar shards.
    
    Usage:
        reader = TarShardReader("data/shards")
        for sample in reader:
            process(sample)
    """
    
    def __init__(self, shard_dir: str, pattern: str = "*.tar"):
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob(pattern))
    
    def __iter__(self):
        for shard_path in self.shards:
            yield from self._read_shard(shard_path)
    
    def _read_shard(self, shard_path: Path):
        """Yield samples from a single shard."""
        # Group files by sample_id
        samples_data = {}
        
        with tarfile.open(shard_path, "r:*") as tar:
            for member in tar.getmembers():
                # Parse filename: "000042.density.npy" -> ("000042", "density", "npy")
                parts = member.name.split(".")
                if len(parts) < 2:
                    continue
                
                sample_id = parts[0]
                field_type = parts[1]
                
                if sample_id not in samples_data:
                    samples_data[sample_id] = {}
                
                # Read file content
                f = tar.extractfile(member)
                if f is None:
                    continue
                
                if field_type == "json":
                    samples_data[sample_id]["metadata"] = json.load(f)
                else:
                    # Load numpy array
                    buf = io.BytesIO(f.read())
                    samples_data[sample_id][field_type] = np.load(buf)
        
        # Yield samples
        for sample_id in sorted(samples_data.keys()):
            data = samples_data[sample_id]
            yield Sample(
                sample_id=sample_id,
                density=data.get("density"),
                displacement=data.get("displacement"),
                stress_vm=data.get("stress"),
                input_tensor=data.get("input"),
                metadata=data.get("metadata")
            )
    
    def __len__(self):
        """Approximate count (requires reading all shards)."""
        return len(self.shards) * 1000  # Estimate
