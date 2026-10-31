"""Verify the downloaded official checkpoint, strict EMA loading, optionally inference.

Run in the dedicated Python 3.11 environment. Checkpoint pickle loading is limited
to the exact official artifact SHA256 recorded below. No dataset is needed.
"""
import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "0fe2d385d074834f3c22d9458798da49b2e5a58bd817284f130aae8814f5c1ef"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    report = {"status": "failed", "python": sys.version, "platform": platform.platform(),
              "device_requested": args.device, "forward_requested": args.forward}
    output = ROOT / "reports" / "backbone_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = ROOT / "checkpoints" / "backbone.ckpt"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["checkpoint_sha256"] = digest
        if digest != EXPECTED_SHA256:
            raise RuntimeError("Checkpoint hash differs from the recorded official download")
        import torch
        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not count this as GPU validation")
        if args.device == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
        sys.path.insert(0, str(ROOT / "vendor" / "EEGDM"))
        from model.diffusion_model import Wavenet
        # Trusted official checkpoint, hash-checked above; contains Lightning metadata.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        kwargs = checkpoint["hyper_parameters"]["model_kwargs"]
        report["model_kwargs"] = kwargs
        backbone = Wavenet(**kwargs)
        prefix = "ema.ema_model."
        state = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items()
                 if k.startswith(prefix)}
        if not state:
            raise RuntimeError("Expected EMA backbone parameters were not found")
        result = backbone.load_state_dict(state, strict=True)
        report.update(missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys,
                      parameter_count=sum(p.numel() for p in backbone.parameters()))
        backbone.requires_grad_(False).eval().to(args.device)
        report["trainable_parameters"] = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        if args.forward:
            from model.classifier import LatentActivityExtractor, LatentActivityReducer
            # FP1-F3 is condition 14 in this pinned upstream channel order.
            extractor = LatentActivityExtractor(backbone, diffusion_t=1, query=["gate"], use_cond=[14]).eval().to(args.device)
            report["forward"] = []
            torch.manual_seed(0)
            with torch.inference_mode():
                for length in (1000, 6000):
                    reducer = LatentActivityReducer(query=["gate"], reduce=["std"], L=length,
                                                   window_size=200, window_step=200).eval().to(args.device)
                    x = torch.randn(1, 1, length, device=args.device) * 0.1
                    features = reducer(extractor((x, None)))
                    finite = bool(torch.isfinite(features).all().item())
                    if not finite:
                        raise RuntimeError("Nonfinite extracted features")
                    report["forward"].append({"input_shape": list(x.shape),
                                              "feature_shape": list(features.shape), "finite": finite})
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
