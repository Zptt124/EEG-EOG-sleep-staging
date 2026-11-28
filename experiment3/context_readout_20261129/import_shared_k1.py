"""Import the 260 equivalent C fits into the locked D training output directory."""

import argparse
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def import_shared(args):
    joint_sha, cache_d = digest(args.lock), digest(args.d_cache)
    archive_sha = digest(args.archive)
    audit_sha = digest(args.input_audit)
    lock = json.loads(args.lock.read_bytes())
    audit = json.loads(args.input_audit.read_bytes())
    cache_c = lock["sha256"]["cache_C"]
    if (set(lock["approved_by"]) != {"C", "D"}
            or not all(item["approved"] is True for item in lock["approved_by"].values())
            or lock["sha256"]["cache_D"] != cache_d
            or archive_sha != args.archive_sha256
            or lock["approved_by"]["C"]["verified_shared_K1_inputs"] != audit
            or audit["input_C_sha256"] != cache_c or audit["input_D_sha256"] != cache_d
            or audit["historical_RP_units_exact"] != 260 or audit["evaluation_centers_exact"] != 41498):
        raise ValueError("The C archive and equivalent K1 inputs must match the joint approval")
    with np.load(args.d_cache, allow_pickle=False) as data:
        units = {(int(r), int(d), str(s)) for r, d, s in zip(data["roster"], data["draw"], data["strategy"])
                 if 1 <= d <= 5 and s in ("add_people", "random")}
    if len(units) != 260:
        raise ValueError("D input lacks the complete shared R/P draws 1–5")

    def output_stem(unit):
        r, d, s = unit
        return f"{r:02d}_{d:02d}_{s}_k1_seed0"

    files = {output_stem(unit) + suffix for unit in units for suffix in (".npz", ".pt")}
    args.output.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        if set(archive.namelist()) != files or len(archive.namelist()) != len(files):
            raise ValueError("Unexpected shared K1 archive inventory")
        planned = []
        for unit in sorted(units):
            basename = output_stem(unit)
            target = args.output / (basename + ".npz")
            model = target.with_suffix(".pt")
            if target.exists():
                with np.load(target, allow_pickle=False) as prior:
                    if ("shared_K1_archive_sha256" not in prior
                            or prior["input_sha256"].item() != cache_d or prior["joint_sha256"].item() != joint_sha
                            or prior["shared_K1_archive_sha256"].item() != archive_sha
                            or not model.is_file() or prior["model_sha256"].item() != digest(model)):
                        raise ValueError(f"Refusing to replace an existing unrelated fit: {target}")
                continue
            planned.append((unit, basename, target, model))
        for unit, basename, target, model in planned:
            data = archive.read(basename + ".npz")
            weights = archive.read(basename + ".pt")
            with np.load(BytesIO(data), allow_pickle=False) as raw:
                if ((raw["roster"].item(), raw["draw"].item(), raw["strategy"].item()) != unit
                        or (raw["k"].item(), raw["seed"].item(), raw["balanced"].item()) != (1, 0, False)
                        or raw["input_sha256"].item() != cache_c or raw["joint_sha256"].item() != joint_sha
                        or sha256(weights).hexdigest() != raw["model_sha256"].item()
                        or raw["pred"].shape != (41498,) or raw["counts"].shape != (16, 5, 5)):
                    raise ValueError(f"Shared fit payload differs from the C lock: {basename}")
                payload = {name: raw[name] for name in raw.files}
            if model.exists() and digest(model) != payload["model_sha256"].item():
                raise ValueError(f"Refusing to replace an existing unrelated model: {model}")
            # D binds the imported result to the equivalent local input, while the
            # archived original and fit_input_sha256 retain the input used to fit.
            payload["input_sha256"] = cache_d
            payload["fit_input_sha256"] = cache_c
            payload["shared_K1_archive_sha256"] = archive_sha
            payload["shared_K1_equivalence_sha256"] = audit_sha
            with target.with_suffix(".npz.partial").open("wb") as file:
                np.savez_compressed(file, **payload)
            model.with_suffix(".pt.partial").write_bytes(weights)
            model.with_suffix(".pt.partial").replace(model)
            target.with_suffix(".npz.partial").replace(target)
    print(f"Ready: {len(units)} shared R/P fits, {len(planned)} imported this time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "input-audit", "lock", "d-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    import_shared(parser.parse_args())
