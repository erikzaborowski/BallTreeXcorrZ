"""Shared count-product identity checks."""
import re


from .counts_io import CountProductError, canonical_json


def _same(a, b, message):
    if canonical_json(a) != canonical_json(b):
        raise CountProductError(message)


def _catalogue(run, side):
    """A digest of actual coordinates/weights/cells, plus saved cell moments."""
    digest = run.metadata.get("catalogue_hashes", {}).get(side)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CountProductError("reconstruction requires per-side catalogue hashes")
    sums = getattr(run.counts, "weight_sums_" + side)
    moments = getattr(run, side + "_moments")
    return {"sha256": digest, "cells": list(sums), "moments": moments.tolist()}
