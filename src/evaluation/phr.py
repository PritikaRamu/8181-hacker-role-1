from dataclasses import dataclass
from typing import List, Dict, Any
import os
import sys
import json

LIST_PACKAGE_FILES = {
    "Python": "data/package_list/pypi_package_names.txt",
    "JavaScript": "data/package_list/npm_package_names.txt",
    "Ruby": "data/package_list/rubygems_package_names.txt",
    "Rust": "data/package_list/cargo_package_names.txt",
}

@dataclass
class PHR:
    language: str
    response: List[str]
    validation: List[bool]
    total_generated_packages: int
    total_valid_packages: int
    phr_score: float

def load_package_list(language: str) -> List[str]:
    """
        Load the package list for the given language.
        Args:
            language: The language of the package list.
        Returns:
            List[str]: The package list.
    """
    with open(LIST_PACKAGE_FILES[language], "r", encoding="utf-8") as f:
        package_list = f.read().splitlines()
    return package_list

def validate_response(response: List[str], package_list: List[str]) -> List[bool]:
    """
        Validate the response against the package list. If the response is in the package list, it is considered valid.
        Args:
            response: The response to validate.
            package_list: The package list to validate against.
        Returns:
            List[bool]: The validation results.
    """
    return [package in package_list for package in response]

def calculate_total_packages(response: List[str]) -> int:
    """
        Calculate the total number of packages in the response.
        Args:
            response: The response to calculate the total number of packages from.
        Returns:
            int: The total number of packages.
    """
    return len(response)

def calculate_total_hallucinations(validation: List[bool]) -> int:
    """
        Calculate the total number of hallucinations in the validation results.
        Args:
            validation: The validation results.
        Returns:
            int: The total number of hallucinations.
    """
    return sum(validation)

def calculate_phr(language: str, response: List[str], package_list: List[str] = None) -> PHR:
    """
        Calculate the PHR score for the given response.
        Args:
            language: The language of the response.
            response: The response to calculate the PHR score for.
            package_list: The package list to validate against. If None, the package list will be loaded from the default location.
        Returns:
            PHR: The PHR score.
    """
    if package_list is None:
        package_list = load_package_list(language)
    validation = validate_response(response, package_list)
    total_generated_packages = calculate_total_packages(response)
    total_valid_packages = calculate_total_packages(validation)
    phr_score = total_valid_packages / total_generated_packages
    return PHR(language, response, validation, total_generated_packages, total_valid_packages, phr_score)

