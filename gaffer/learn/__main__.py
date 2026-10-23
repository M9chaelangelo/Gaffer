"""`python -m gaffer.learn` still trains, as the workflow expects.

With names after it, only those models are refitted and the rest are left on
disk with the scores they already had:

    python -m gaffer.learn ceiling
"""
import sys

from .train import train

if __name__ == "__main__":
    train(only=sys.argv[1:] or None)
