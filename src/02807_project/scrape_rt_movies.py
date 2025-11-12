"""Web scraper for Rotten Tomatoes movie titles, descriptions, and release years.

This script scrapes movie details from Rotten Tomatoes using the movie IDs
from the rotten_tomatoes_critic_reviews dataset. It uses concurrent workers
with rate limiting and resumability support.
"""

import asyncio
import csv
from pathlib import Path
from typing import TypedDict

import httpx
import polars as pl
import requests
from helpers.logger import logger
from lxml import html
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Configuration
RAW_LOCATION = Path("data/raw")
INPUT_FILE = RAW_LOCATION / "rotten_tomatoes_critic_reviews.csv"
OUTPUT_FILE = RAW_LOCATION / "rotten_tomatoes_movie_details.csv"
PROGRESS_FILE = RAW_LOCATION / ".scrape_progress.txt"
BASE_URL = "https://www.rottentomatoes.com/"

# Scraping parameters
MAX_WORKERS = 10
REQUEST_DELAY = 0.5  # seconds between requests per worker (start aggressive)
MAX_RETRIES = 3
TIMEOUT = 30.0  # seconds
HTTP_TOO_MANY_REQUESTS = 429  # Rate limit status code

# Year validation constants
YEAR_LENGTH = 4
MIN_YEAR = 1900
MAX_YEAR = 2030


class MovieDetails(TypedDict):
    """Type definition for scraped movie details."""

    rotten_tomatoes_link: str
    title: str | None
    description: str | None
    release_year: str | None
    scrape_status: str


class RateLimiter:
    """Simple rate limiter using asyncio."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_request
            if time_since_last < self.delay:
                await asyncio.sleep(self.delay - time_since_last)
            self._last_request = asyncio.get_event_loop().time()


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(MAX_RETRIES),
)
async def fetch_movie_page(client: httpx.AsyncClient, movie_id: str, rate_limiter: RateLimiter) -> str:
    """Fetch the HTML content of a movie page with retry logic.

    Args:
        client: HTTP client
        movie_id: Movie ID (e.g., "m/0814255")
        rate_limiter: Rate limiter instance

    Returns:
        HTML content as string

    Raises:
        httpx.HTTPError: If request fails after retries

    """
    await rate_limiter.acquire()

    url = f"{BASE_URL}{movie_id}"
    response = await client.get(url, timeout=TIMEOUT, follow_redirects=True)

    # Handle rate limiting
    if response.status_code == requests.codes.too_many_requests:
        retry_after = int(response.headers.get("Retry-After", "60"))
        logger.warning(f"Rate limited! Waiting {retry_after} seconds...")
        await asyncio.sleep(retry_after)
        response.raise_for_status()

    response.raise_for_status()
    return response.text


def extract_movie_details(html_content: str, movie_id: str) -> MovieDetails:
    """Extract title, description, and release year from HTML using XPath.

    Args:
        html_content: Raw HTML content
        movie_id: Movie ID for reference

    Returns:
        Dictionary with movie details

    """
    try:
        tree = html.fromstring(html_content)

        # Extract title
        title_elements = tree.xpath("/html/body/div[3]/main/div/div[1]/div[1]/div/media-hero/rt-text[2]//text()")
        title = " ".join(title_elements).strip() if title_elements else None

        # Extract description
        desc_elements = tree.xpath(
            "/html/body/div[3]/main/div/div[1]/div[2]/div[1]/div[1]/media-scorecard/div[1]/drawer-more/rt-text//text()"
        )
        description = " ".join(desc_elements).strip() if desc_elements else None

        # Extract release year - try multiple approaches to avoid picking up runtime
        release_year = None

        # First try: Look for year patterns in media-hero rt-text elements
        hero_texts = tree.xpath("/html/body/div[3]/main/div/div[1]/div[1]/div/media-hero//rt-text//text()")
        for hero_text in hero_texts:
            cleaned_text = hero_text.strip()
            # Look for 4-digit years (19xx or 20xx)
            if (
                cleaned_text.isdigit()
                and len(cleaned_text) == YEAR_LENGTH
                and MIN_YEAR <= int(cleaned_text) <= MAX_YEAR
            ):
                release_year = cleaned_text
                break

        # Second try: Look for year in score details or other metadata areas
        if not release_year:
            score_details = tree.xpath("//score-details//text()")
            for score_text in score_details:
                cleaned_text = score_text.strip()
                if (
                    cleaned_text.isdigit()
                    and len(cleaned_text) == YEAR_LENGTH
                    and MIN_YEAR <= int(cleaned_text) <= MAX_YEAR
                ):
                    release_year = cleaned_text
                    break

        # Third try: Look in the breadcrumb or metadata sections
        if not release_year:
            breadcrumb_texts = tree.xpath("//breadcrumb//text() | //metadata//text()")
            for breadcrumb_text in breadcrumb_texts:
                cleaned_text = breadcrumb_text.strip()
                if (
                    cleaned_text.isdigit()
                    and len(cleaned_text) == YEAR_LENGTH
                    and MIN_YEAR <= int(cleaned_text) <= MAX_YEAR
                ):
                    release_year = cleaned_text
                    break

    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to parse HTML for {movie_id}: {e}")
        return {
            "rotten_tomatoes_link": movie_id,
            "title": None,
            "description": None,
            "release_year": None,
            "scrape_status": f"parse_error: {e!s}",
        }
    return {
        "rotten_tomatoes_link": movie_id,
        "title": title,
        "description": description,
        "release_year": release_year,
        "scrape_status": "success",
    }


async def scrape_movie(
    client: httpx.AsyncClient,
    movie_id: str,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
) -> MovieDetails:
    """Scrape a single movie's details.

    Args:
        client: HTTP client
        movie_id: Movie ID to scrape
        rate_limiter: Rate limiter instance
        semaphore: Semaphore for worker limit

    Returns:
        Movie details dictionary

    """
    async with semaphore:
        try:
            html_content = await fetch_movie_page(client, movie_id, rate_limiter)
            return extract_movie_details(html_content, movie_id)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {movie_id}: {e.response.status_code}")
            return {
                "rotten_tomatoes_link": movie_id,
                "title": None,
                "description": None,
                "release_year": None,
                "scrape_status": f"http_error_{e.response.status_code}",
            }
        except httpx.TimeoutException:
            logger.error(f"Timeout for {movie_id}")
            return {
                "rotten_tomatoes_link": movie_id,
                "title": None,
                "description": None,
                "release_year": None,
                "scrape_status": "timeout",
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error for {movie_id}: {e}")
            return {
                "rotten_tomatoes_link": movie_id,
                "title": None,
                "description": None,
                "release_year": None,
                "scrape_status": f"error: {e!s}",
            }


def load_scraped_movies() -> set[str]:
    """Load already scraped movie IDs from output file."""
    if not OUTPUT_FILE.exists():
        return set()

    try:
        df = pl.read_csv(OUTPUT_FILE)
        return set(df["rotten_tomatoes_link"].to_list())
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not load existing scraped data: {e}")
        return set()


def save_results(results: list[MovieDetails], mode: str = "a") -> None:
    """Save scraping results to CSV.

    Args:
        results: List of movie details
        mode: File mode ('w' for write, 'a' for append)

    """
    if not results:
        return

    file_exists = OUTPUT_FILE.exists() and mode == "a"

    with OUTPUT_FILE.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rotten_tomatoes_link", "title", "description", "release_year", "scrape_status"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    logger.info(f"💾 Saved {len(results)} results to {OUTPUT_FILE}")


async def scrape_all_movies(movie_ids: list[str]) -> None:
    """Scrape all movies with concurrent workers.

    Args:
        movie_ids: List of unique movie IDs to scrape

    """
    rate_limiter = RateLimiter(REQUEST_DELAY)
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    # Initialize output file if it doesn't exist
    if not OUTPUT_FILE.exists():
        save_results([], mode="w")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",  # noqa: E501
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    async with httpx.AsyncClient(headers=headers) as client:
        batch_size = 100  # Save progress every 100 movies
        results_batch = []

        tasks = [scrape_movie(client, movie_id, rate_limiter, semaphore) for movie_id in movie_ids]

        total = len(movie_ids)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results_batch.append(result)
            completed += 1  # noqa: SIM113

            # Log progress
            if completed % 10 == 0 or completed == total:
                success_count = sum(1 for r in results_batch if r["scrape_status"] == "success")
                logger.info(
                    f"Progress: {completed}/{total} ({completed / total * 100:.1f}%) - {success_count} successful"
                )

            # Save batch periodically
            if len(results_batch) >= batch_size:
                save_results(results_batch, mode="a")
                results_batch = []

        # Save remaining results
        if results_batch:
            save_results(results_batch, mode="a")


def get_unique_movie_ids() -> list[str]:
    """Load unique movie IDs from the input CSV, excluding already scraped ones.

    Returns:
        List of unique movie IDs to scrape

    """
    if not INPUT_FILE.exists():
        msg = f"Input file not found: {INPUT_FILE}"
        raise FileNotFoundError(msg)

    logger.info(f"📂 Loading movie IDs from {INPUT_FILE}")

    # Load all unique movie IDs
    df = pl.scan_csv(INPUT_FILE, schema_overrides={"rotten_tomatoes_link": pl.Utf8})
    unique_ids = df.select(pl.col("rotten_tomatoes_link").unique()).collect()["rotten_tomatoes_link"].to_list()

    logger.info(f"Found {len(unique_ids)} unique movie IDs")

    # Load already scraped IDs
    scraped_ids = load_scraped_movies()
    logger.info(f"Already scraped: {len(scraped_ids)} movies")

    # Filter out already scraped
    remaining_ids = [movie_id for movie_id in unique_ids if movie_id not in scraped_ids]

    logger.info(f"Remaining to scrape: {len(remaining_ids)} movies")

    return remaining_ids


async def main() -> None:
    """Entry point for the scraper."""
    logger.info("🚀 Starting Rotten Tomatoes scraper")
    logger.info(f"Configuration: {MAX_WORKERS} workers, {REQUEST_DELAY}s delay per worker")

    try:
        movie_ids = get_unique_movie_ids()

        if not movie_ids:
            logger.info("✅ No movies to scrape - all done!")
            return

        await scrape_all_movies(movie_ids)

        logger.info("🎉 Scraping completed!")

        # Load and display summary
        if OUTPUT_FILE.exists():
            df = pl.read_csv(OUTPUT_FILE)
            total = len(df)
            successful = len(df.filter(pl.col("scrape_status") == "success"))
            logger.info(
                f"📊 Summary: {successful}/{total} movies scraped successfully ({successful / total * 100:.1f}%)"
            )

    except KeyboardInterrupt:
        logger.warning("⚠️  Scraping interrupted by user")
        logger.info("Progress has been saved. Run again to resume.")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
