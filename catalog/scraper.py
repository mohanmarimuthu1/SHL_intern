"""
Scrapes the SHL Individual Test Solutions catalog and saves to catalog.json.
Run once to build the data layer: python -m catalog.scraper
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_test_types(soup: BeautifulSoup) -> list[str]:
    """Extract test type codes from a product page."""
    types = []
    # SHL uses colored circles/labels for test types
    for tag in soup.find_all(class_=re.compile(r"product-catalogue.*type|test.?type", re.I)):
        text = tag.get_text(strip=True)
        for code in TEST_TYPE_MAP:
            if code in text:
                types.append(code)
    return list(set(types)) or []


def parse_product_details(url: str) -> dict:
    """Scrape individual product page for details."""
    soup = get_page(url)
    if not soup:
        return {}

    details = {}

    # Description — usually in a <p> or .product-description block
    desc_tag = (
        soup.find(class_=re.compile(r"product.?desc|overview|about", re.I))
        or soup.find("meta", {"name": "description"})
    )
    if desc_tag:
        if desc_tag.name == "meta":
            details["description"] = desc_tag.get("content", "").strip()
        else:
            details["description"] = desc_tag.get_text(" ", strip=True)[:800]

    # Duration
    duration_tag = soup.find(string=re.compile(r"minutes|duration|timing", re.I))
    if duration_tag:
        m = re.search(r"(\d+)\s*min", str(duration_tag), re.I)
        if m:
            details["duration_minutes"] = int(m.group(1))

    # Remote testing / adaptive flags
    page_text = soup.get_text(" ", strip=True).lower()
    details["remote_testing"] = any(
        kw in page_text for kw in ["remote testing", "remotely proctored", "online administration"]
    )
    details["adaptive_irt"] = any(
        kw in page_text for kw in ["adaptive", "irt", "item response theory"]
    )

    # Test types from detail page
    types = parse_test_types(soup)
    if types:
        details["test_types"] = types

    return details


def scrape_catalog() -> list[dict]:
    """
    Scrape the SHL product catalog table.
    The catalog uses a filterable table with pagination.
    Individual Test Solutions are filtered by type.
    """
    assessments = []
    seen_urls = set()

    # The catalog page uses query params for filtering & pagination
    # type=1 corresponds to Individual Test Solutions
    page = 0
    per_page = 12  # SHL default page size

    print("Scraping SHL catalog...")

    while True:
        params = f"?start={page * per_page}&type=1"
        url = CATALOG_URL + params
        print(f"  Fetching page {page + 1}: {url}")

        soup = get_page(url)
        if not soup:
            print("  Failed to fetch page, stopping.")
            break

        # Find the catalog table rows
        rows = soup.select("table.product-catalogue tbody tr, .product-catalogue__row, [class*='catalogue'] tr")

        # Fallback: find all links that look like product pages
        if not rows:
            rows = soup.select(".product-catalogue-table tr")

        if not rows:
            # Try finding product cards/links directly
            product_links = soup.select("a[href*='/solutions/products/']")
            if not product_links:
                print(f"  No products found on page {page + 1}, stopping.")
                break

            for link in product_links:
                href = link.get("href", "")
                if not href or "product-catalog" in href:
                    continue
                full_url = BASE_URL + href if href.startswith("/") else href
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)
                name = link.get_text(strip=True)
                if name:
                    assessments.append({"name": name, "url": full_url})
        else:
            found_new = False
            for row in rows:
                cols = row.find_all(["td", "th"])
                if not cols or len(cols) < 2:
                    continue

                # First col: product name + link
                name_col = cols[0]
                link_tag = name_col.find("a")
                if not link_tag:
                    continue

                href = link_tag.get("href", "")
                full_url = BASE_URL + href if href.startswith("/") else href
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)
                found_new = True

                name = link_tag.get_text(strip=True)

                # Extract test type dots/labels from remaining columns
                test_types = []
                for col in cols[1:]:
                    col_text = col.get_text(strip=True)
                    # Check for checkmarks or type codes
                    if col_text in TEST_TYPE_MAP:
                        test_types.append(col_text)
                    # Check for filled circles (SHL uses • or similar markers)
                    if col.find(class_=re.compile(r"yes|check|filled|active", re.I)):
                        # Need header context — handled below
                        pass

                entry = {
                    "name": name,
                    "url": full_url,
                    "test_types": test_types,
                }
                assessments.append(entry)

            if not found_new:
                print(f"  No new products on page {page + 1}, stopping.")
                break

        page += 1
        time.sleep(1)  # polite crawl delay

        # Safety cap
        if page > 50:
            break

    return assessments


def enrich_with_details(assessments: list[dict]) -> list[dict]:
    """Visit each product page to fill in description, duration, etc."""
    print(f"\nEnriching {len(assessments)} assessments with detail pages...")
    enriched = []
    for i, item in enumerate(assessments):
        print(f"  [{i+1}/{len(assessments)}] {item['name']}")
        details = parse_product_details(item["url"])
        enriched.append({**item, **details})
        time.sleep(0.8)
    return enriched


def main():
    assessments = scrape_catalog()
    print(f"\nFound {len(assessments)} assessments in catalog listing.")

    if not assessments:
        print("ERROR: No assessments scraped. Check SHL site structure.")
        return

    # Enrich with detail pages
    assessments = enrich_with_details(assessments)

    # Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(assessments)} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
