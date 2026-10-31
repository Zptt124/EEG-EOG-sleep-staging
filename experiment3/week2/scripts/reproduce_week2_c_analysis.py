"""Replay W2 metrics and error analysis into a new directory; no model fitting."""
import argparse
from pathlib import Path
import analyze_week2_c_results as analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    target=(analysis.ROOT/args.output).resolve()
    if not target.is_relative_to(analysis.ROOT.resolve()) or target.exists():
        raise ValueError('Use a new output directory inside experiment3')
    analysis.OUT=target
    analysis.analyze()


if __name__=='__main__':main()
