"""
Rechunk and optimize HDF5 files for random-access DataLoader performance.


This script creates optimized copies with:
  - Row-aligned chunks (e.g. 1024 rows × all columns) for fast single-row random access
  - Blosc/LZ4 compression (fast decompression, decent ratio)
  - float32 downcast for predictors (matches what the model uses)
  - Larger chunks for range mask datasets

Usage:
    python scripts/data_prep_scripts/rechunk_h5.py [--data-dir /path/to/data] [--dry-run]
"""

import argparse
import os
import shutil
import time
import h5py
import numpy as np
import yaml

# Try to use hdf5plugin for blosc/lz4 (much faster decompression than gzip)
try:
    import hdf5plugin
    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False
    print("WARNING: hdf5plugin not installed. Will use gzip instead of blosc/lz4.")
    print("         Install with: pip install hdf5plugin")
    print("         Blosc/LZ4 is ~5-10x faster to decompress than gzip.\n")


def get_compression_kwargs():
    """Return the best available compression settings.

    NOTE: If using Blosc, you MUST `import hdf5plugin` before reading the files.
    """
    if BLOSC_AVAILABLE:
        print("  [compression] Using Blosc/LZ4 (requires `import hdf5plugin` at read time)")
        return hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    else:
        print("  [compression] Using gzip level 1 (no extra dependencies needed)")
        return {"compression": "gzip", "compression_opts": 1}  # fast gzip


def rechunk_predictor_file(src_path, dst_path, chunk_rows=1024, dry_run=False):
    """
    Rechunk a predictor HDF5 file:
      - Chunk shape: (chunk_rows, ncols) for fast random row access
      - Downcast float64 → float32
      - Add compression
    """
    with h5py.File(src_path, "r") as f_in:
        dset = f_in["predictor_data"]
        shape = dset.shape
        dtype_in = dset.dtype
        chunks_in = dset.chunks
        compression_in = dset.compression

    dtype_out = np.float32
    nrows, ncols = shape
    chunk_shape = (min(chunk_rows, nrows), ncols)

    size_in = os.path.getsize(src_path) / (1024 ** 2)
    est_size_out = (nrows * ncols * 4) / (1024 ** 2)  # float32, before compression

    print(f"\n  {os.path.basename(src_path)}")
    print(f"    shape={shape}, dtype: {dtype_in}→{dtype_out}")
    print(f"    chunks: {chunks_in} → {chunk_shape}")
    print(f"    compression: {compression_in} → {'blosc/lz4' if BLOSC_AVAILABLE else 'gzip'}")
    print(f"    size: {size_in:.1f} MB → ~{est_size_out:.1f} MB (uncompressed f32)")

    if dry_run:
        return

    comp_kwargs = get_compression_kwargs()

    t0 = time.time()
    with h5py.File(src_path, "r") as f_in, h5py.File(dst_path, "w") as f_out:
        dset_in = f_in["predictor_data"]

        if isinstance(comp_kwargs, dict):
            dset_out = f_out.create_dataset(
                "predictor_data",
                shape=shape,
                dtype=dtype_out,
                chunks=chunk_shape,
                **comp_kwargs,
            )
        else:
            # hdf5plugin filter object
            dset_out = f_out.create_dataset(
                "predictor_data",
                shape=shape,
                dtype=dtype_out,
                chunks=chunk_shape,
                **comp_kwargs,
            )

        # Copy in batches to keep memory reasonable
        batch_size = max(chunk_rows * 64, 65536)  # ~64 chunks per batch
        for start in range(0, nrows, batch_size):
            end = min(start + batch_size, nrows)
            data = dset_in[start:end].astype(np.float32)
            dset_out[start:end] = data
            if (start // batch_size) % 10 == 0:
                print(f"    ... {start:,}/{nrows:,} rows", end="\r")

    elapsed = time.time() - t0
    size_out = os.path.getsize(dst_path) / (1024 ** 2)
    print(f"    Done in {elapsed:.1f}s. Output: {size_out:.1f} MB (ratio: {size_in / max(size_out, 0.1):.1f}x)")


def rechunk_range_mask_file(src_path, dst_path, dry_run=False):
    """
    Rechunk a range mask HDF5 file with better chunk sizes.
    Original has chunks=(1,) on cumulative_lengths.
    """
    with h5py.File(src_path, "r") as f_in:
        datasets = {}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    "shape": obj.shape,
                    "dtype": obj.dtype,
                    "chunks": obj.chunks,
                    "compression": obj.compression,
                }
        f_in.visititems(visitor)

    print(f"\n  {os.path.basename(src_path)}")
    for name, info in datasets.items():
        n = info["shape"][0]
        # Use ~256KB chunks (good balance for random and sequential)
        bytes_per_elem = np.dtype(info["dtype"]).itemsize
        target_chunk_bytes = 256 * 1024  # 256 KB
        new_chunk_size = max(1024, target_chunk_bytes // bytes_per_elem)
        new_chunk_size = min(new_chunk_size, n)
        print(f"    {name}: shape={info['shape']}, chunks: {info['chunks']} → ({new_chunk_size},)")

    if dry_run:
        return

    comp_kwargs = get_compression_kwargs()
    t0 = time.time()

    with h5py.File(src_path, "r") as f_in, h5py.File(dst_path, "w") as f_out:
        # Copy attributes
        for key, val in f_in.attrs.items():
            f_out.attrs[key] = val

        for name in datasets:
            dset_in = f_in[name]
            n = dset_in.shape[0]
            dtype = dset_in.dtype
            bytes_per_elem = dtype.itemsize
            target_chunk_bytes = 256 * 1024
            new_chunk_size = max(1024, target_chunk_bytes // bytes_per_elem)
            new_chunk_size = min(new_chunk_size, n)

            if isinstance(comp_kwargs, dict):
                dset_out = f_out.create_dataset(
                    name, shape=(n,), dtype=dtype,
                    chunks=(new_chunk_size,), **comp_kwargs,
                )
            else:
                dset_out = f_out.create_dataset(
                    name, shape=(n,), dtype=dtype,
                    chunks=(new_chunk_size,), **comp_kwargs,
                )

            batch_size = max(new_chunk_size * 64, 1_000_000)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                dset_out[start:end] = dset_in[start:end]
                if (start // batch_size) % 10 == 0:
                    print(f"    {name}: {start:,}/{n:,}", end="\r")
            print(f"    {name}: done                    ")

    elapsed = time.time() - t0
    size_in = os.path.getsize(src_path) / (1024 ** 2)
    size_out = os.path.getsize(dst_path) / (1024 ** 2)
    print(f"    Done in {elapsed:.1f}s. {size_in:.1f} MB → {size_out:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Rechunk HDF5 files for fast random access")
    # Default the data root to whatever configs/local_default.yaml points at.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(_root, "configs", "local_default.yaml")) as _f:
            _default_data_dir = yaml.safe_load(_f).get("data_dir")
    except (FileNotFoundError, KeyError, TypeError):
        _default_data_dir = None
    parser.add_argument("--data-dir", default=_default_data_dir, required=_default_data_dir is None,
                        help="Root data directory (default: data_dir from configs/local_default.yaml)")
    parser.add_argument("--chunk-rows", type=int, default=1024,
                        help="Number of rows per chunk for predictor files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be done, don't create files")
    parser.add_argument("--skip-predictors", action="store_true",
                        help="Skip rechunking predictor files")
    parser.add_argument("--skip-masks", action="store_true",
                        help="Skip rechunking range mask files")
    parser.add_argument("--skip-species-index", action="store_true",
                        help="Skip rechunking species index files")
    parser.add_argument("--in-place", action="store_true",
                        help="Replace original files (moves original to .bak first)")
    args = parser.parse_args()

    data_dir = args.data_dir
    pred_dir = os.path.join(data_dir, "predictors")
    mask_dirs = [
        os.path.join(data_dir, "range_masks"),
        os.path.join(data_dir, "range_masks_1km"),
    ]

    # --- Predictor files ---
    if not args.skip_predictors:
        print("=" * 60)
        print("RECHUNKING PREDICTOR FILES")
        print("=" * 60)

        pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".h5"))
        for fname in pred_files:
            src = os.path.join(pred_dir, fname)
            dst = os.path.join(pred_dir, fname.replace(".h5", "_rechunked.h5"))

            # Check if already well-chunked
            with h5py.File(src, "r") as f:
                dset = f["predictor_data"]
                if dset.chunks is not None and dset.dtype == np.float32:
                    print(f"\n  SKIP {fname} (already chunked + float32)")
                    continue

            rechunk_predictor_file(src, dst, chunk_rows=args.chunk_rows, dry_run=args.dry_run)

            if args.in_place and not args.dry_run:
                bak = src + ".bak"
                print(f"    Moving {fname} → {fname}.bak")
                shutil.move(src, bak)
                shutil.move(dst, src)

    # --- Range mask files ---
    if not args.skip_masks:
        print("\n" + "=" * 60)
        print("RECHUNKING RANGE MASK FILES")
        print("=" * 60)

        for mask_dir in mask_dirs:
            if not os.path.isdir(mask_dir):
                continue
            mask_files = sorted(f for f in os.listdir(mask_dir)
                                if f.endswith(".h5") and "range_mask" in f)
            for fname in mask_files:
                src = os.path.join(mask_dir, fname)
                dst = os.path.join(mask_dir, fname.replace(".h5", "_rechunked.h5"))

                # Check if already has reasonable chunks
                needs_rechunk = False
                with h5py.File(src, "r") as f:
                    for name in f:
                        if isinstance(f[name], h5py.Dataset):
                            if f[name].chunks is not None and f[name].chunks[0] <= 1:
                                needs_rechunk = True
                                break
                if not needs_rechunk:
                    print(f"\n  SKIP {os.path.relpath(src, data_dir)} (chunks OK)")
                    continue

                rechunk_range_mask_file(src, dst, dry_run=args.dry_run)

                if args.in_place and not args.dry_run:
                    bak = src + ".bak"
                    print("    Moving original → .bak")
                    shutil.move(src, bak)
                    shutil.move(dst, src)

    # --- Species index files ---
    if not args.skip_species_index:
        print("\n" + "=" * 60)
        print("RECHUNKING SPECIES INDEX FILES")
        print("=" * 60)

        for mask_dir in mask_dirs:
            if not os.path.isdir(mask_dir):
                continue
            species_files = sorted(f for f in os.listdir(mask_dir)
                                   if f.endswith(".h5") and "species_index" in f)
            for fname in species_files:
                src = os.path.join(mask_dir, fname)
                dst = os.path.join(mask_dir, fname.replace(".h5", "_rechunked.h5"))

                # Check if already using blosc or has large enough chunks
                needs_rechunk = False
                with h5py.File(src, "r") as f:
                    for name in f:
                        if isinstance(f[name], h5py.Dataset):
                            d = f[name]
                            plist = d.id.get_create_plist()
                            nf = plist.get_nfilters()
                            uses_blosc = False
                            for i in range(nf):
                                info = plist.get_filter(i)
                                if info[3] and b'blosc' in info[3]:
                                    uses_blosc = True
                            if not uses_blosc:
                                needs_rechunk = True
                                break
                if not needs_rechunk:
                    print(f"\n  SKIP {os.path.relpath(src, data_dir)} (already blosc)")
                    continue

                rechunk_range_mask_file(src, dst, dry_run=args.dry_run)

                if args.in_place and not args.dry_run:
                    bak = src + ".bak"
                    print("    Moving original → .bak")
                    shutil.move(src, bak)
                    shutil.move(dst, src)

    print("\n" + "=" * 60)
    print("DONE!")
    if not args.in_place and not args.dry_run:
        print("Rechunked files saved with '_rechunked.h5' suffix.")
        print("To replace originals, re-run with --in-place")
    print("=" * 60)


if __name__ == "__main__":
    main()
