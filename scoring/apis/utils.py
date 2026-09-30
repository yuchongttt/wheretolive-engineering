#!/usr/bin/env python3
"""API parameter loading helper."""

import json
import os
from typing import Dict, Any


def load_api_params(params_file: str = "property_params.json") -> Dict[str, Any]:
    """
    Load API parameter configuration.

    Secrets are never read from the file: if the GOOGLE_API_KEY environment
    variable is set it is injected as params["google_api"]["api_key"], which is
    what the evaluators check to enable the (paid) Google Routes/Places/Geocoding
    APIs. Without it they fall back to the free TfL / postcodes.io paths.

    Args:
        params_file: path to the parameter file (see property_params.example.json)

    Returns:
        Parameter dict
    """
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Parameter file {params_file} not found")
        params = {}
    except json.JSONDecodeError as e:
        print(f"Error parsing parameter file: {e}")
        params = {}

    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        params.setdefault("google_api", {})["api_key"] = google_key
    return params
