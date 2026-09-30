#!/usr/bin/env python3
"""UK Land Registry SPARQL API wrapper"""

import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from apis.api_usage import record_api_usage


class LandRegistryAPI:
    """UK Land Registry SPARQL API client wrapper"""

    def __init__(self):
        """
        Initialise the Land Registry API
        This API is free and needs no authentication
        """
        self.endpoint = "http://landregistry.data.gov.uk/landregistry/query"

    def _normalize_postcode(self, postcode: str) -> str:
        """
        Normalise the postcode format

        Args:
            postcode: raw postcode string

        Returns:
            Normalised postcode (upper case, format "XX## #XX")
        """
        # Remove all spaces and upper-case
        cleaned = postcode.replace(" ", "").upper()

        # UK postcode format: first part 2-4 chars, second part 3 chars
        # Insert a space before the third-to-last character
        if len(cleaned) > 3:
            return f"{cleaned[:-3]} {cleaned[-3:]}"
        return cleaned

    def _build_sparql_query(self, postcode: str, years: int = 10) -> str:
        """
        Build the SPARQL query

        Args:
            postcode: normalised postcode
            years: how many years of data to query

        Returns:
            SPARQL query string
        """
        # Compute the start date
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        query = f"""
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>
PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>

SELECT ?paon ?saon ?street ?price ?date ?propertyType ?estateType ?newBuild ?transactionCategory
WHERE {{
  ?transaction lrppi:pricePaid ?price ;
               lrppi:transactionDate ?date ;
               lrppi:propertyAddress ?addr .

  ?addr lrcommon:postcode "{postcode}"^^xsd:string .

  OPTIONAL {{ ?addr lrcommon:street ?street . }}
  OPTIONAL {{ ?addr lrcommon:paon ?paon . }}
  OPTIONAL {{ ?addr lrcommon:saon ?saon . }}

  OPTIONAL {{ ?transaction lrppi:propertyType ?propertyType . }}
  OPTIONAL {{ ?transaction lrppi:estateType ?estateType . }}
  OPTIONAL {{ ?transaction lrppi:newBuild ?newBuild . }}
  OPTIONAL {{ ?transaction lrppi:transactionCategory ?transactionCategory . }}

  FILTER (?date >= "{start_date}"^^xsd:date)
}}
ORDER BY DESC(?date)
LIMIT 2000
"""
        return query

    def get_transactions_by_postcode(
        self,
        postcode: str,
        years: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch property transactions for a postcode

        Args:
            postcode: postcode
            years: how many years of data to query (default 30)
            max_retries: maximum number of retries
            retry_delay: retry delay (seconds)

        Returns:
            List of transactions, or None on failure
        """
        normalized_postcode = self._normalize_postcode(postcode)
        query = self._build_sparql_query(normalized_postcode, years)

        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "query": query
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    record_api_usage("land_registry")
                    results = response.json()
                    bindings = results.get("results", {}).get("bindings", [])

                    transactions = []
                    for binding in bindings:
                        # Parse the property type
                        property_type_uri = binding.get("propertyType", {}).get("value", "")
                        property_type = self._parse_property_type(property_type_uri)

                        # Parse the tenure type
                        estate_type_uri = binding.get("estateType", {}).get("value", "")
                        tenure_type = self._parse_estate_type(estate_type_uri)

                        # Build the address
                        paon = binding.get("paon", {}).get("value", "")
                        saon = binding.get("saon", {}).get("value", "")
                        street = binding.get("street", {}).get("value", "")

                        address_parts = []
                        if saon:
                            address_parts.append(saon)
                        if paon:
                            address_parts.append(paon)
                        if street:
                            address_parts.append(street)

                        address = ", ".join(address_parts) if address_parts else "Unknown"

                        # Parse newBuild
                        new_build_value = binding.get("newBuild", {}).get("value", "")
                        new_build = new_build_value.lower() == "true" if new_build_value else False

                        # Parse the transaction category URI → 'A' (standard market sale) / 'B' (additional price
                        # paid: company purchases / repossessions / power-of-sale and other non-standard
                        # transfers). Missing → None, which consumers treat as A, same as category IS NULL
                        # in the local DB.
                        category_uri = binding.get("transactionCategory", {}).get("value", "")
                        if "additionalPricePaid" in category_uri:
                            category = "B"
                        elif "standardPricePaid" in category_uri:
                            category = "A"
                        else:
                            category = None

                        transaction = {
                            "address": address,
                            "postcode": normalized_postcode,
                            "price": int(float(binding.get("price", {}).get("value", 0))),
                            "date": binding.get("date", {}).get("value", ""),
                            "property_type": property_type,
                            "tenure_type": tenure_type,
                            "new_build": new_build,
                            "category": category
                        }
                        transactions.append(transaction)

                    if attempt > 0:
                        print(f"  ✅ Land Registry API: attempt {attempt + 1} succeeded")

                    return transactions

                elif response.status_code == 429:
                    last_error = "Land Registry API rate limited (429)"
                    wait_time = retry_delay * (attempt + 1) * 2
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue

                elif response.status_code >= 500:
                    last_error = f"Land Registry API server error ({response.status_code})"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue

                else:
                    last_error = f"Land Registry API request failed: {response.status_code} - {response.text[:200]}"
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"  ❌ {last_error}")
                        return None

            except requests.exceptions.Timeout:
                last_error = "Land Registry API request timed out"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"Land Registry API request error: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                last_error = f"Error processing Land Registry response: {str(e)[:100]}"
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {last_error}; retrying in {retry_delay}s ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    continue

        print(f"  ❌ Land Registry API still failing after {max_retries} retries: {last_error}")
        return None

    def _parse_property_type(self, uri: str) -> str:
        """Parse a property-type URI into a short code"""
        type_mapping = {
            "detached": "D",
            "semi-detached": "S",
            "terraced": "T",
            "flat-maisonette": "F",
            "other": "O"
        }
        for key, code in type_mapping.items():
            if key in uri.lower():
                return code
        return "O"

    def _parse_estate_type(self, uri: str) -> str:
        """Parse an estate-type (tenure) URI into a short code"""
        if "freehold" in uri.lower():
            return "F"
        elif "leasehold" in uri.lower():
            return "L"
        return "U"
