"""
scripts/setup_nltk.py
Run once after pip install to download required NLTK corpora.
Usage: python scripts/setup_nltk.py
"""

import nltk

packages = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
]

for pkg in packages:
    print(f"Downloading NLTK package: {pkg}")
    nltk.download(pkg, quiet=True)

print("\n✅  NLTK setup complete.")