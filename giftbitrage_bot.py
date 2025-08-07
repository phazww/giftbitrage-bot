"""
giftbitrage_bot.py
====================

This module contains a complete example of a Telegram bot that scans two
popular Telegram‑based NFT gift marketplaces (Tonnel and Portals) to find
arbitrage opportunities.  The bot uses the open source `portalsmp` and
`tonnelmp` libraries to communicate with the respective marketplaces and
implements a simple profit‑calculation algorithm.  It allows users to
configure price and profit thresholds and responds to a `/scan` command by
returning only those gifts that meet the user’s criteria.

**How it works**
----------------

1.  Upon startup the bot reads configuration from a `.env` file in the
    working directory.  You should set the following variables:

    - `BOT_TOKEN`: Your Telegram bot token obtained from @BotFather.
    - `PORTALS_API_ID` and `PORTALS_API_HASH`: Credentials for
      authorising with the Telegram API to obtain an `authData` token for
      Portals via `portalsmp.update_auth`.
    - `TONNEL_AUTH_DATA`: A token string for Tonnel obtained by logging in
      through a browser and extracting `web-initData` from local storage.

2.  The bot defines a data class `GiftCandidate` to hold information
    about a potential arbitrage opportunity, including buy/sell prices,
    profit amount and percentage, and whether the gift is "clean".

3.  When the `/start` command is received, the bot greets the user and
    briefly explains what it can do.

4.  The `/scan` command triggers the scanning process.  It accepts up
    to three optional parameters: minimum price, maximum price and
    minimum profit percentage.  If omitted, default values are used.

5.  The scanning routine performs the following steps:

    - It ensures a fresh `authData` token is available for Portals by
      calling `update_auth`.  Since this call is asynchronous, it uses
      `asyncio.run` internally.
    - It fetches floor prices for all gifts from Portals via
      `giftsFloors`.  This returns a mapping from short gift names to
      their lowest sale price on Portals.
    - It fetches pages of active gifts for sale on Tonnel via
      `getGifts`.  The price range filter is applied on the server so
      that only gifts within the specified bounds are returned.  The
      scan continues until either there are no more gifts or a
      preset limit of pages is reached (to avoid excessive API
      requests).
    - For each gift on Tonnel, the bot normalises its name by removing
      spaces, hyphens and punctuation and converting it to lower case.
      This normalised name is then used to look up floor prices on
      Portals.  If a floor price exists there and is higher than the
      Tonnel price, an arbitrage candidate is created with the
      corresponding buy and sell information.
    - The profit is calculated both as an absolute difference and a
      percentage of the buy price.  A constant commission (5%) is
      subtracted from the difference when the buy and sell markets are
      different.  Candidates not meeting the minimum profit percentage
      are discarded.
    - The status of the gift as "clean" or "dirty" is inferred from
      the `signature` field returned by Tonnel's API.  If signature is
      missing or empty, the gift is considered clean.

6.  The bot formats the list of arbitrage candidates into a user‑friendly
    message.  Each entry shows the gift name, direction of the trade,
    current price, potential second price, profit in TON and percentage,
    and whether the gift is clean or dirty.  If no opportunities are
    found, the user is informed accordingly.

This example is designed for demonstration purposes.  It can be
extended by adding more sophisticated heuristics (e.g. analysing
historical sale data or activity over time), additional markets, or a
cron‑like scheduler to perform scans periodically.  The use of
asynchronous code ensures that network requests do not block the bot
while it interacts with users.
"""

from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command, CommandObject
from aiogram.exceptions import TelegramNetworkError
from dotenv import load_dotenv

# External marketplace modules
# We wrap the imports in a try/except so that if the packages are missing,
# a helpful error message is raised.  Both import lines must be inside
# the try block; otherwise Python will expect an 'except' after the first
try:
    # Import Portals and Tonnel helpers.  We also import the top‑level
    # tonnelmp module so that functions such as tonnelmp.getAuctions are
    # available.  If any of these imports fail (e.g. missing package),
    # we fall back to the ImportError handler below.
    from portalsmp import update_auth, giftsFloors, search
    from tonnelmp import getGifts, filterStatsPretty
    import tonnelmp  # needed for tonnelmp.getAuctions in auction scanning
except ImportError:
    raise ImportError(
        "Required packages 'portalsmp' and 'tonnelmp' are not installed. "
        "Install them via pip: pip install portalsmp tonnelmp"
    )

# Commission and fee settings.
#
# Both marketplaces charge a commission on trades.  When you purchase a gift,
# you pay the listing price plus the commission of the market where you buy;
# when you sell, you receive the listing price minus the commission of the
# market where you sell.  Additionally, moving a gift from one marketplace
# to another incurs a fixed fee (on-chain transfer cost), typically around
# 0.1–0.2 TON.  These constants can be tuned if fee structures change.
COMMISSION_RATE_TONNEL = 0.06  # 6% commission on purchases/sales at Tonnel
COMMISSION_RATE_PORTALS = 0.05  # 5% commission on purchases/sales at Portals

# Fixed fee (in TON) for transferring a gift between markets.  When buying
# on one marketplace and selling on another, this cost is subtracted from
# the profit.  Adjust this value if transfer fees change.
TRANSFER_FEE = 0.15

# ---------------------------------------------------------------------------
# Proxy configuration
#
# To improve reliability when calling the Tonnel APIs (which are protected by
# Cloudflare), you can specify an outbound proxy via environment variables.
# The bot will construct a proxy URL from the following variables if they
# exist:
#   PROXY_IP:       the IP address or hostname of your proxy server
#   PROXY_PORT:     the proxy port
#   PROXY_LOGIN:    (optional) username for proxy authentication
#   PROXY_PASSWORD: (optional) password for proxy authentication
#   PROXY_URL:      alternatively, a full proxy URL can be provided.  If
#                   PROXY_URL is set, it takes precedence over the
#                   decomposed variables above.
#
# Example in your .env file:
#   PROXY_IP=91.216.186.249
#   PROXY_PORT=8000
#   PROXY_LOGIN=LghvXN
#   PROXY_PASSWORD=7W2268
#
# With these values the bot will construct:
#   http://LghvXN:7W2268@91.216.186.249:8000
# and use it for both HTTP and HTTPS requests.  If no proxy variables are
# provided, no proxy will be used.

# Build a proxies dictionary from environment variables.  This is evaluated
# at import time so that the same proxy settings are reused throughout
# execution.  If you change proxy settings, restart the bot.
_proxy_url_env = os.getenv("PROXY_URL")
if _proxy_url_env:
    PROXIES: Optional[Dict[str, str]] = {"http": _proxy_url_env, "https": _proxy_url_env}
else:
    _proxy_ip = os.getenv("PROXY_IP")
    _proxy_port = os.getenv("PROXY_PORT")
    if _proxy_ip and _proxy_port:
        _proxy_login = os.getenv("PROXY_LOGIN")
        _proxy_password = os.getenv("PROXY_PASSWORD")
        if _proxy_login and _proxy_password:
            _proxy_url = f"http://{_proxy_login}:{_proxy_password}@{_proxy_ip}:{_proxy_port}"
        else:
            _proxy_url = f"http://{_proxy_ip}:{_proxy_port}"
        PROXIES = {"http": _proxy_url, "https": _proxy_url}
    else:
        PROXIES = None

# ---------------------------------------------------------------------------
# Cloudflare clearance cookie configuration
#
# The Tonnel marketplace is protected by Cloudflare.  If your IP address
# becomes blocked or you are presented with a challenge page, you can
# bypass this protection by providing a valid `__cf_clearance` cookie from
# your browser.  To use this feature set `CF_CLEARANCE` in your `.env`
# file to the long token value copied from your browser's cookies (for
# example: CF_CLEARANCE=TxVC9HWglApX8UnAfPRlJCS1Sf7LZ_AvFU05WEmJteI-...).
# If present, this cookie will be added to the headers of every request
# made by the underlying `tonnelmp` library, which can help Cloudflare
# accept your requests.

_cf_clearance = os.getenv("CF_CLEARANCE")
if _cf_clearance:
    try:
        import tonnelmp.marketapi as _tm_marketapi  # type: ignore
        # Inject the clearance cookie into the shared HEADERS used by
        # tonnelmp.  All requests will include this cookie header.
        _tm_marketapi.HEADERS["cookie"] = f"__cf_clearance={_cf_clearance}"
    except Exception as exc:
        # If the module import fails, silently ignore; the library may
        # not be installed yet.  We avoid raising here to allow the bot
        # to continue loading, but without the Cloudflare cookie the
        # requests may be blocked.
        print(f"Warning: unable to set CF_CLEARANCE cookie: {exc}")

@dataclass
class GiftCandidate:
    """Representation of a single arbitrage opportunity."""

    name: str
    model: str
    backdrop: str
    symbol: str
    price_buy: float
    price_sell: float
    profit_absolute: float
    profit_percent: float
    market_buy: str  # "Tonnel" or "Portals"
    market_sell: str
    clean: bool


def normalise_name(name: str) -> str:
    """Convert a gift name into a canonical form for matching across markets.

    Spaces, hyphens and apostrophes are removed and the result is
    lowercased.  This function mirrors the behaviour of the helper
    `toShortName` in the portalsmp codebase.
    """
    import re
    short = re.sub(r"[\s\-'’]", "", name).lower()
    return short


async def fetch_portals_floors(auth_data: str) -> Dict[str, float]:
    """Return a mapping of short gift names to floor prices from Portals.

    Parameters
    ----------
    auth_data: str
        The authentication token retrieved via update_auth.

    Returns
    -------
    Dict[str, float]
        Mapping from normalised gift name to the minimum sell price on
        Portals.
    """
    floors = giftsFloors(auth_data)
    # The API should return a list of dicts.  If it returns a string or any
    # other type (e.g. due to auth failure), skip processing and return
    # an empty mapping.
    result: Dict[str, float] = {}
    # When giftsFloors returns a string or empty response, log and return empty.
    if not floors or isinstance(floors, str):
        print(f"Received unexpected Portals floors response: {floors}")
        return result
    # The portalsmp.giftsFloors function returns either a dict mapping
    # gift short names to price strings or a list of dicts with keys
    # 'name' and 'price'.  Handle both formats gracefully.
    # If floors is a dictionary: keys are short names, values are prices.
    if isinstance(floors, dict):
        for name, price_str in floors.items():
            if price_str is None:
                continue
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                continue
            result[normalise_name(name)] = price
        return result
    # Otherwise treat floors as an iterable of dicts.  Skip unexpected types.
    for item in floors:
        if not isinstance(item, dict):
            print(
                f"Skipping unexpected floor item of type {type(item)}: {item}"  # nosec B005
            )
            continue
        name = item.get("name")
        price_str = item.get("price")
        if not name or price_str is None:
            continue
        try:
            price = float(price_str)
        except (ValueError, TypeError):
            continue
        result[normalise_name(name)] = price
    return result


async def fetch_tonnel_gifts(
    auth_data: str,
    min_price: float,
    max_price: float,
    max_pages: int = 20,
) -> List[Dict]:
    """Retrieve all active gifts from Tonnel within a price range.

    This function paginates through the available gifts on Tonnel.  It
    stops when either `max_pages` pages have been loaded or the API
    returns fewer items than expected (indicating the end of the list).

    Parameters
    ----------
    auth_data: str
        The authentication token for Tonnel (web-initData value).
    min_price: float
        Minimum price in TON for gifts to include.
    max_price: float
        Maximum price in TON for gifts to include.
    max_pages: int
        Maximum number of pages to request.

    Returns
    -------
    List[Dict]
        List of gift dictionaries as returned by tonnelmp.getGifts.
    """
    all_gifts: List[Dict] = []
    page = 1
    while page <= max_pages:
        # price_range expects a two element list [min, max]
        try:
            # Pass the proxy settings through to the Tonnel API call.  The
            # tonnelmp.getGifts function accepts a 'proxies' argument
            # compatible with curl_cffi.requests.  Using a proxy can help
            # bypass Cloudflare filtering on gifts2.tonnel.network.
            gifts_page = getGifts(
                limit=30,
                page=page,
                sort="price_asc",
                price_range=[min_price, max_price],
                asset="TON",
                authData=auth_data,
                proxies=PROXIES,
            )
        except Exception as exc:
            # Log the error and break the loop.  In a production bot
            # consider retrying or notifying an administrator.
            print(f"Error fetching page {page}: {exc}")
            break
        if not gifts_page:
            # If the API returned an empty response or a non-list, stop scanning.
            break
        # If the API returns a string or other non-list, abort scanning to avoid
        # iterating over a string later.  This can happen if the auth token is
        # invalid or Cloudflare blocks the request.
        if isinstance(gifts_page, str):
            print(
                f"Received unexpected string instead of list on page {page}: {gifts_page[:80]}"
            )
            break
        all_gifts.extend(gifts_page)
        # If fewer items than limit, we've reached the end.
        if len(gifts_page) < 30:
            break
        page += 1
    return all_gifts


async def fetch_tonnel_floors(auth_data: str) -> Dict[str, float]:
    """Return a mapping of short gift names to floor prices on Tonnel.

    This helper uses the `filterStatsPretty` API, which returns
    aggregated statistics for each gift.  The returned structure
    contains, for each gift name, a "data" section with the lowest
    floor price across all models, as well as counts.  We flatten
    this into a simple dict of { normalised_gift_name: floor_price }.

    Parameters
    ----------
    auth_data: str
        Authentication token for Tonnel (web-initData value).

    Returns
    -------
    Dict[str, float]
        Mapping from normalised gift name to floor price.  Gifts with
        missing or null floor price are skipped.
    """
    result: Dict[str, float] = {}
    try:
        # Pass proxy settings to filterStatsPretty.  Using a proxy can
        # help get around Cloudflare blocking when fetching stats from
        # gifts3.tonnel.network.
        stats = filterStatsPretty(auth_data, proxies=PROXIES)
    except Exception as exc:
        print(f"Failed to fetch Tonnel floors: {exc}")
        return result
    # Expect stats to be a dict with keys 'status' and 'data'
    if not isinstance(stats, dict):
        print(f"Unexpected Tonnel floor response: {stats}")
        return result
    data = stats.get("data")
    if not isinstance(data, dict):
        print(f"Unexpected Tonnel floor data: {data}")
        return result
    for gift_key, gift_data in data.items():
        # gift_key is already lowercased in filterStatsPretty
        floor_info = gift_data.get("data", {})
        floor_price = floor_info.get("floorPrice")
        if floor_price is None:
            continue
        try:
            price = float(floor_price)
        except (TypeError, ValueError):
            continue
        result[normalise_name(gift_key)] = price
    return result

async def fetch_tonnel_auctions(
    auth_data: str,
    min_price: float,
    max_price: float,
    max_pages: int = 3,
) -> List[Dict]:
    """Retrieve active auctions from Tonnel within a price range.

    The Tonnel marketplace exposes auctions via the same `pageGifts` endpoint as
    regular listings, filtered by the presence of an ``auction_id``.  This
    helper wraps the ``tonnelmp.getAuctions`` function and paginates through
    the results until either the specified number of pages have been loaded or
    a request fails (typically due to Cloudflare).

    Parameters
    ----------
    auth_data: str
        The authentication token for Tonnel (``web-initData`` value).
    min_price: float
        Minimum auction price in TON to include.
    max_price: float
        Maximum auction price in TON to include.
    max_pages: int
        Maximum number of pages of auctions to request.  Each page contains
        up to 30 auctions.  A low value helps avoid Cloudflare bans.

    Returns
    -------
    List[Dict]
        A list of auction objects returned by the API.  Each dict contains
        details such as gift name, model, current bid and auction end time.
    """
    auctions: List[Dict] = []
    page = 1
    # Tonnel API expects price_range as either an integer or a two‑element list
    if max_price == float('inf'):
        price_range: list | int = int(min_price)
    else:
        price_range = [int(min_price), int(max_price)]
    while page <= max_pages:
        try:
            page_auctions = tonnelmp.getAuctions(
                page=page,
                limit=30,
                price_range=price_range,
                authData=auth_data,
                proxies=PROXIES,
            )
        except Exception as exc:
            # If Cloudflare blocks the request or any other error occurs,
            # stop pagination and return what has been collected so far.
            print(f"Error fetching auctions page {page}: {exc}")
            break
        if not page_auctions or not isinstance(page_auctions, list):
            break
        auctions.extend(page_auctions)
        # Stop if the returned page contains fewer than the limit (end of list)
        if len(page_auctions) < 30:
            break
        page += 1
    return auctions

async def fetch_portals_model_prices(
    auth_data: str,
    model_keys: List[Tuple[str, str]],
    min_price: float,
    max_price: float,
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Return floor and second floor prices on Portals for specific gift‑model pairs.

    For each (gift, model) tuple in ``model_keys``, this helper calls
    ``portalsmp.search`` with parameters to fetch the two cheapest listings on
    Portals.  It then returns a mapping from the tuple to a tuple of
    ``(floor_price, second_price)``.  Prices that fall outside the
    ``min_price``/``max_price`` bounds are ignored (set to ``None``).

    Parameters
    ----------
    auth_data: str
        Portals authentication token (obtained via ``update_auth``).
    model_keys: List[Tuple[str, str]]
        List of tuples ``(gift_name, model_name)`` for which to fetch prices.
    min_price: float
        Minimum price to include in the results.  Listings cheaper than this
        will be treated as missing.
    max_price: float
        Maximum price to include in the results.  Listings more expensive
        will be treated as missing.

    Returns
    -------
    Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]
        Mapping from ``(gift_name, model_name)`` to a two‑element tuple
        ``(floor_price, second_price)``.  Missing or out‑of‑range values are
        represented as ``None``.
    """
    results: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {}
    for gift_name, model_name in model_keys:
        try:
            # Normalise gift name to match Portals API expectations
            short_name = gift_name
            # The search function accepts either str or list for gift_name/model
            listings = search(
                sort="price_asc",
                offset=0,
                limit=2,
                gift_name=short_name,
                model=model_name,
                authData=auth_data,
            )
        except Exception as exc:
            print(f"Error fetching Portals listings for {gift_name} {model_name}: {exc}")
            results[(gift_name, model_name)] = (None, None)
            continue
        floor_price: Optional[float] = None
        second_price: Optional[float] = None
        # search returns a list of dicts; each dict has a 'price' field
        if isinstance(listings, list):
            # iterate over results and pick first two within price range
            prices: List[float] = []
            for item in listings:
                if not isinstance(item, dict):
                    continue
                price_str = item.get("price")
                if price_str is None:
                    continue
                try:
                    price = float(price_str)
                except (TypeError, ValueError):
                    continue
                if price < min_price or price > max_price:
                    continue
                prices.append(price)
            if prices:
                floor_price = prices[0]
                if len(prices) > 1:
                    second_price = prices[1]
        results[(gift_name, model_name)] = (floor_price, second_price)
    return results

def calculate_auction_flips(
    auctions: List[Dict],
    portals_prices: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    min_profit_percent: float,
    clean_status: Dict[str, bool],
) -> List[GiftCandidate]:
    """Compute arbitrage opportunities from Tonnel auctions to Portals.

    Given a list of auctions (active on Tonnel) and the corresponding per‑model
    floor and second prices on Portals, compute the potential profit for
    buying a gift on auction and selling it on Portals.  A sale price is
    chosen as the second cheapest listing on Portals when available; if
    only one listing exists, the floor price is used.  The profit is
    calculated as:

      profit = sell_price * (1 - COMMISSION_RATE_PORTALS) - buy_price * (1 + COMMISSION_RATE_TONNEL) - TRANSFER_FEE

    Only profits meeting ``min_profit_percent`` are returned.

    Parameters
    ----------
    auctions: List[Dict]
        The active auctions retrieved from Tonnel.
    portals_prices: Dict[Tuple[str,str],Tuple[Optional[float],Optional[float]]]
        Mapping from (gift, model) to a tuple of (floor_price, second_price)
        on Portals.
    min_profit_percent: float
        Minimum percentage profit required for an arbitrage opportunity.
    clean_status: Dict[str, bool]
        Mapping from normalised gift names to a boolean indicating whether
        the gift is "clean" (no signature) based on scanned Tonnel pages.

    Returns
    -------
    List[GiftCandidate]
        List of profitable arbitrage candidates.
    """
    opportunities: List[GiftCandidate] = []
    for auction in auctions:
        if not isinstance(auction, dict):
            continue
        gift_name = auction.get("gift_name") or auction.get("name")
        model_name = auction.get("model")
        if not gift_name or not model_name:
            continue
        # Determine current bid price.  The API may provide a 'highestBid' dict
        # or 'price' field for auctions.  Fallback to 'price' if no bids.
        buy_price = None
        # Check for 'highestBid' dict
        highest_bid = auction.get("highestBid")
        if isinstance(highest_bid, dict):
            amount = highest_bid.get("amount")
            if amount is not None:
                try:
                    buy_price = float(amount)
                except (TypeError, ValueError):
                    pass
        # Fallback to 'price' field (starting price)
        if buy_price is None:
            price_str = auction.get("price") or auction.get("startPrice")
            if price_str is not None:
                try:
                    buy_price = float(price_str)
                except (TypeError, ValueError):
                    buy_price = None
        if buy_price is None:
            continue
        # Retrieve Portals floor and second floor prices for this model
        prices = portals_prices.get((gift_name, model_name))
        if not prices:
            continue
        floor_price, second_price = prices
        # Use second_price if available; otherwise use floor_price
        sell_price = second_price if second_price is not None else floor_price
        if sell_price is None:
            continue
        # Compute gross profit: revenue minus costs and fees
        revenue = sell_price * (1 - COMMISSION_RATE_PORTALS)
        cost = buy_price * (1 + COMMISSION_RATE_TONNEL)
        profit = revenue - cost - TRANSFER_FEE
        if profit <= 0:
            continue
        profit_percent = (profit / cost) * 100.0
        if profit_percent < min_profit_percent:
            continue
        # Determine cleanliness based on gift name
        short_name = normalise_name(gift_name)
        clean = clean_status.get(short_name, True)
        candidate = GiftCandidate(
            name=gift_name,
            model=model_name,
            backdrop=auction.get("backdrop") or "",
            symbol=auction.get("symbol") or "",
            price_buy=buy_price * (1 + COMMISSION_RATE_TONNEL),
            price_sell=sell_price,
            profit_absolute=profit,
            profit_percent=profit_percent,
            market_buy="Tonnel (auction)",
            market_sell="Portals",
            clean=clean,
        )
        opportunities.append(candidate)
    return opportunities

def calculate_portals_internal_flips(
    portals_prices: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
    min_profit_percent: float,
) -> List[GiftCandidate]:
    """Compute arbitrage opportunities within Portals for each model.

    Given floor and second floor prices for each (gift, model) pair on
    Portals, compute the profit from buying at the floor price and selling
    at the second price, after accounting for Portals commission on both
    the purchase and the sale.  Only opportunities meeting the minimum
    profit percentage are returned.

    Parameters
    ----------
    portals_prices: Dict[Tuple[str,str],Tuple[Optional[float],Optional[float]]]
        Mapping from (gift, model) to (floor_price, second_price).
    min_profit_percent: float
        Minimum profit percentage required.

    Returns
    -------
    List[GiftCandidate]
        List of profitable internal Portals arbitrage opportunities.
    """
    opportunities: List[GiftCandidate] = []
    for (gift_name, model_name), (floor_price, second_price) in portals_prices.items():
        if floor_price is None or second_price is None:
            continue
        # Buy at floor, sell at second price, pay commission on both sides
        revenue = second_price * (1 - COMMISSION_RATE_PORTALS)
        cost = floor_price * (1 + COMMISSION_RATE_PORTALS)
        profit = revenue - cost
        if profit <= 0:
            continue
        profit_percent = (profit / cost) * 100.0
        if profit_percent < min_profit_percent:
            continue
        candidate = GiftCandidate(
            name=gift_name,
            model=model_name,
            backdrop="",
            symbol="",
            price_buy=cost,
            price_sell=second_price,
            profit_absolute=profit,
            profit_percent=profit_percent,
            market_buy="Portals",
            market_sell="Portals",
            clean=True,
        )
        opportunities.append(candidate)
    return opportunities


async def fetch_tonnel_model_floors(
    auth_data: str,
    min_price: float | None = None,
    max_price: float | None = None,
) -> Dict[tuple[str, str], float]:
    """
    Collect floor prices on Tonnel at the granularity of gift model.

    This helper wraps the `filterStatsPretty` API to return a mapping
    from (gift_name, model_name) to the cheapest price found for that
    model.  It will filter entries by the supplied price range.

    Parameters
    ----------
    auth_data: str
        Authentication token for Tonnel (web-initData value).
    min_price: float | None
        Minimum acceptable price for the model floor.  Models with
        price strictly below this value will be ignored.  If None,
        no lower bound is applied.
    max_price: float | None
        Maximum acceptable price for the model floor.  Models with
        price strictly above this value will be ignored.  If None,
        no upper bound is applied.

    Returns
    -------
    Dict[(str, str), float]
        Mapping from tuples of (normalized gift name, normalized model name)
        to the floor price (float) for that specific model.  Models
        whose floor price is missing or cannot be parsed are skipped.
    """
    result: Dict[tuple[str, str], float] = {}
    try:
        stats = filterStatsPretty(auth_data, proxies=PROXIES)
    except Exception as exc:
        print(f"Failed to fetch Tonnel model floors: {exc}")
        return result
    if not isinstance(stats, dict):
        return result
    data = stats.get("data")
    if not isinstance(data, dict):
        return result
    # Normalize min and max
    low = min_price if min_price is not None else 0.0
    high = max_price if max_price is not None else float("inf")
    for gift_key, gift_data in data.items():
        if not isinstance(gift_data, dict):
            continue
        # iterate models within the gift
        for model_key, model_info in gift_data.items():
            # skip the aggregated 'data' entry
            if model_key == "data":
                continue
            if not isinstance(model_info, dict):
                continue
            floor_price = model_info.get("floorPrice")
            if floor_price is None:
                continue
            try:
                price = float(floor_price)
            except (TypeError, ValueError):
                continue
            # apply price range filter
            if price < low or price > high:
                continue
            gift_norm = normalise_name(gift_key)
            model_norm = normalise_name(model_key)
            # record the minimum price for this gift-model pair
            current = result.get((gift_norm, model_norm))
            if current is None or price < current:
                result[(gift_norm, model_norm)] = price
    return result


async def fetch_portals_model_floors(
    auth_data: str,
    model_keys: list[tuple[str, str]],
    min_price: float | None = None,
    max_price: float | None = None,
) -> Dict[tuple[str, str], float]:
    """
    Query Portals for the cheapest listing of each (gift, model) pair.

    This helper iterates over the supplied list of gift-model tuples
    and for each performs a search on Portals using filters for gift
    name and model name.  It returns the lowest price found.  A
    price range may be supplied to filter results.

    Parameters
    ----------
    auth_data: str
        Authentication token for Portals (tma token returned by update_auth).
    model_keys: list[tuple[str, str]]
        List of (gift_name, model_name) pairs (both normalized) for
        which to fetch floor prices.
    min_price: float | None
        Minimum acceptable price.  If provided, models cheaper than
        this value on Portals are ignored.  Note that this check is
        applied after retrieving the price and does not affect the
        search query.
    max_price: float | None
        Maximum acceptable price.  If provided, models more expensive
        than this value on Portals are ignored.

    Returns
    -------
    Dict[(str, str), float]
        Mapping from (gift_name, model_name) to the lowest found
        listing price on Portals.  Models for which no listing is
        found are omitted from the returned dict.
    """
    result: Dict[tuple[str, str], float] = {}
    # normalise price limits
    low = min_price if min_price is not None else 0.0
    high = max_price if max_price is not None else float("inf")
    for gift_name, model_name in model_keys:
        try:
            # Portals API expects capitalisation; use cap() helper.
            # We search with limit=1 to get only the cheapest listing.
            data = search(
                sort="price_asc",
                offset=0,
                limit=1,
                gift_name=gift_name,
                model=model_name,
                authData=auth_data,
            )
        except Exception as exc:
            # ignore individual search failures
            continue
        # search() may return either a list of dicts or a dict with key "results"
        listings: list[dict] | None = None
        if isinstance(data, list):
            listings = data
        elif isinstance(data, dict):
            listings = data.get("results") if isinstance(data.get("results"), list) else None
        if not listings:
            continue
        listing = listings[0]
        price_str = listing.get("price")
        if price_str is None:
            continue
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            continue
        # apply price range
        if price < low or price > high:
            continue
        result[(gift_name, model_name)] = price
    return result


def calculate_model_flips(
    tonnel_model_floors: Dict[tuple[str, str], float],
    portals_model_floors: Dict[tuple[str, str], float],
    min_profit_percent: float,
    clean_status: Dict[str, bool] | None = None,
) -> List[GiftCandidate]:
    """
    Compute arbitrage opportunities at the gift-model level.

    This routine compares floor prices per model across Tonnel and
    Portals.  For each (gift, model) pair that exists in both
    dictionaries, it determines the cheaper market to buy and the
    more expensive market to sell.  It then calculates the cost
    including commissions, the proceeds after commissions, subtracts
    transfer fees when crossing markets, and filters out deals that
    do not meet the specified minimum profit percentage.

    Parameters
    ----------
    tonnel_model_floors: Dict[(str, str), float]
        Mapping from (gift_name, model_name) to floor price on Tonnel.
    portals_model_floors: Dict[(str, str), float]
        Mapping from (gift_name, model_name) to floor price on Portals.
    min_profit_percent: float
        Minimum required profit percentage.
    clean_status: Dict[str, bool] | None
        Optional mapping from gift short name to cleanliness (True if
        clean, False if dirty).  If provided, the cleanliness for
        each model uses the value for the gift (models are assumed
        to share the same cleanliness as their parent gift).  If
        omitted, all models are considered clean.

    Returns
    -------
    List[GiftCandidate]
        Sorted list of arbitrage candidates, highest absolute profit first.
    """
    opportunities: List[GiftCandidate] = []
    # Iterate over intersection of keys for which we have both floors
    for key in set(tonnel_model_floors.keys()) & set(portals_model_floors.keys()):
        gift_name, model_name = key
        price_tonnel = tonnel_model_floors[key]
        price_portal = portals_model_floors[key]
        # skip if either price missing or zero
        if price_tonnel is None or price_portal is None:
            continue
        # Determine buy and sell markets
        if price_tonnel < price_portal:
            price_buy = price_tonnel
            price_sell = price_portal
            market_buy = "Tonnel"
            market_sell = "Portals"
            clean = True
            if clean_status is not None:
                clean = clean_status.get(gift_name, True)
        elif price_portal < price_tonnel:
            # We generally prefer flips from Tonnel to Portals for higher liquidity.
            # If selling on Tonnel, skip to respect liquidity preference.
            continue
        else:
            continue
        # Compute cost and proceeds including commissions
        # Buying on Tonnel includes commission on top of listing price
        cost = price_buy * (1.0 + COMMISSION_RATE_TONNEL)
        # Selling on Portals deducts commission from proceeds
        proceeds = price_sell * (1.0 - COMMISSION_RATE_PORTALS)
        profit_absolute = proceeds - cost
        # Apply transfer fee when buying and selling on different markets
        if market_buy != market_sell:
            profit_absolute -= TRANSFER_FEE
        if cost <= 0:
            continue
        profit_percent = (profit_absolute / cost) * 100.0
        if profit_absolute < 0 or profit_percent < min_profit_percent:
            continue
        opportunities.append(
            GiftCandidate(
                name=gift_name,
                model=model_name,
                backdrop="",
                symbol="",
                price_buy=cost,
                price_sell=price_sell,
                profit_absolute=profit_absolute,
                profit_percent=profit_percent,
                market_buy=market_buy,
                market_sell=market_sell,
                clean=clean,
            )
        )
    opportunities.sort(key=lambda c: c.profit_absolute, reverse=True)
    return opportunities


def calculate_flips(
    tonnel_gifts: List[Dict],
    portal_floors: Dict[str, float],
    min_profit_percent: float,
) -> List[GiftCandidate]:
    """Generate a list of arbitrage opportunities from raw gift data.

    This function now considers only opportunities where the gift is
    bought on Tonnel and sold on Portals.  It is retained for
    backwards compatibility but superseded by calculate_cross_flips().

    Parameters
    ----------
    tonnel_gifts: List[Dict]
        Gifts for sale on Tonnel.
    portal_floors: Dict[str, float]
        Mapping of short gift names to floor prices on Portals.
    min_profit_percent: float
        Minimum percentage profit for a flip to be considered.

    Returns
    -------
    List[GiftCandidate]
        Sorted list of profitable flips, highest absolute profit first.
    """
    opportunities: List[GiftCandidate] = []
    # Build a mapping of clean status for gifts on Tonnel
    clean_status: Dict[str, bool] = {}
    for gift in tonnel_gifts:
        if not isinstance(gift, dict):
            continue
        name = gift.get("name")
        if not name:
            continue
        short_name = normalise_name(name)
        signature = gift.get("signature")
        clean_status.setdefault(short_name, not bool(signature))
    # Create floors for Tonnel based on the cheapest occurrence in tonnel_gifts
    tonnel_floors: Dict[str, float] = {}
    for gift in tonnel_gifts:
        if not isinstance(gift, dict):
            continue
        name = gift.get("name")
        if not name:
            continue
        short_name = normalise_name(name)
        price_str = gift.get("price")
        if price_str is None:
            continue
        try:
            price = float(price_str)
        except ValueError:
            continue
        current = tonnel_floors.get(short_name)
        if current is None or price < current:
            tonnel_floors[short_name] = price
    # Now compute cross-market flips using both floors
    for short_name in set(tonnel_floors.keys()) & set(portal_floors.keys()):
        price_tonnel = tonnel_floors[short_name]
        price_portal = portal_floors[short_name]
        if price_tonnel is None or price_portal is None:
            continue
        # Determine direction
        if price_tonnel < price_portal:
            price_buy = price_tonnel
            price_sell = price_portal
            market_buy = "Tonnel"
            market_sell = "Portals"
            clean = clean_status.get(short_name, True)
        elif price_portal < price_tonnel:
            price_buy = price_portal
            price_sell = price_tonnel
            market_buy = "Portals"
            market_sell = "Tonnel"
            # When buying on Portals we don't know cleanliness; default to True
            clean = True
        else:
            continue
        # Compute net cost, proceeds and profit after commissions.  This
        # function is legacy and does not include transfer fees, but it
        # now accounts for separate commission rates on each marketplace.
        if market_buy == "Tonnel":
            cost = price_buy * (1.0 + COMMISSION_RATE_TONNEL)
        else:
            cost = price_buy * (1.0 + COMMISSION_RATE_PORTALS)
        if market_sell == "Tonnel":
            proceeds = price_sell * (1.0 - COMMISSION_RATE_TONNEL)
        else:
            proceeds = price_sell * (1.0 - COMMISSION_RATE_PORTALS)
        profit_absolute = proceeds - cost
        if cost == 0:
            continue
        profit_percent = (profit_absolute / cost) * 100.0
        if profit_percent < min_profit_percent:
            continue
        candidate = GiftCandidate(
            name=short_name,
            model="",
            backdrop="",
            symbol="",
            price_buy=cost,
            price_sell=price_sell,
            profit_absolute=profit_absolute,
            profit_percent=profit_percent,
            market_buy=market_buy,
            market_sell=market_sell,
            clean=clean,
        )
        opportunities.append(candidate)
    opportunities.sort(key=lambda c: c.profit_absolute, reverse=True)
    return opportunities


def format_candidates_message(candidates: List[GiftCandidate]) -> str:
    """Format a list of GiftCandidate objects into a human friendly message."""
    if not candidates:
        return "😕 Подходящих флипов не найдено. Попробуйте изменить настройки."
    lines = ["🪙 Найденные флипы:"]
    for i, cand in enumerate(candidates, 1):
        status = "чистый" if cand.clean else "грязный"
        lines.append(
            f"{i}. {cand.name} — купить на {cand.market_buy} за {cand.price_buy:.2f} TON, "
            f"продать на {cand.market_sell} за {cand.price_sell:.2f} TON. "
            f"Прибыль: +{cand.profit_absolute:.2f} TON ({cand.profit_percent:.1f}% с учётом комиссии). "
            f"Состояние: {status}."
        )
    return "\n".join(lines)


async def scan_and_find(
    tonnel_auth: str,
    portals_api_id: str,
    portals_api_hash: str,
    min_price: float,
    max_price: float,
    min_profit_percent: float,
) -> List[GiftCandidate]:
    """Perform full scan and return a list of arbitrage opportunities.

    In this flavour of scanning we deliberately avoid using general
    marketplace listings from Tonnel due to the heavy Cloudflare
    protection on the `filterStatsPretty` endpoint.  Instead we
    concentrate on *auctions* hosted on Tonnel, which can still be
    fetched with a valid Cloudflare clearance cookie.  For each
    auctioned gift model we look up the cheapest and second cheapest
    listing on Portals, calculate the potential profit after
    commissions and a fixed transfer fee, and include the deal if it
    meets the requested minimum profit percentage.  In addition we
    compute purely Portals‑internal flips (buy at the floor, sell at
    the second floor) for the same set of gift models.  Cleanliness
    of each gift is inferred from a sample of standard Tonnel
    listings using ``fetch_tonnel_gifts``; auctions themselves do not
    encode signature status.
    """
    # Step 1: obtain a fresh auth token for Portals
    try:
        portals_auth = await update_auth(portals_api_id, portals_api_hash)
    except Exception as exc:
        print(f"Failed to update Portals auth: {exc}")
        raise
    # Step 2: fetch active auctions on Tonnel.  The ``fetch_tonnel_auctions``
    # helper uses the ``getAuctions`` API internally, respects the
    # specified price range and paginates only a limited number of
    # pages to avoid hitting Cloudflare rate limits.  If no auctions
    # are retrieved, return immediately.
    auctions = await fetch_tonnel_auctions(
        tonnel_auth,
        min_price=min_price,
        max_price=max_price,
        max_pages=5,
    )
    if not auctions:
        return []
    # Step 3: build the set of (gift, model) pairs from these auctions.
    model_keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for auction in auctions:
        if not isinstance(auction, dict):
            continue
        gift = auction.get("gift_name") or auction.get("name")
        model = auction.get("model")
        if not gift or not model:
            continue
        g_norm = normalise_name(gift)
        m_norm = normalise_name(model)
        key = (g_norm, m_norm)
        if key not in seen_keys:
            seen_keys.add(key)
            model_keys.append(key)
    if not model_keys:
        return []
    # Step 4: fetch Portals floor and second floor prices for these models.
    portals_prices = await fetch_portals_model_prices(
        portals_auth,
        model_keys,
        min_price=min_price,
        max_price=max_price,
    )
    # Step 5: compute cleanliness for each gift by sampling regular
    # listings from Tonnel.  Auctions do not expose signature status,
    # so we fall back to this method.  We restrict the scan to a
    # reasonable number of pages to reduce API calls.
    tonnel_gifts = await fetch_tonnel_gifts(
        tonnel_auth,
        min_price,
        max_price,
        max_pages=10,
    )
    clean_status: Dict[str, bool] = {}
    for gift in tonnel_gifts:
        if not isinstance(gift, dict):
            continue
        name = gift.get("name")
        if not name:
            continue
        short_name = normalise_name(name)
        signature = gift.get("signature")
        current = clean_status.get(short_name, True)
        clean_status[short_name] = current and not bool(signature)
    # Step 6: compute arbitrage opportunities: (a) Tonnel auctions → Portals;
    # (b) internal Portals flips.  We combine the results and sort by
    # absolute profit descending.
    auction_flips = calculate_auction_flips(
        auctions,
        portals_prices,
        min_profit_percent=min_profit_percent,
        clean_status=clean_status,
    )
    portals_flips = calculate_portals_internal_flips(
        portals_prices,
        min_profit_percent=min_profit_percent,
    )
    all_flips = auction_flips + portals_flips
    all_flips.sort(key=lambda c: c.profit_absolute, reverse=True)
    return all_flips


# Telegram bot setup
router = Router()


@router.message(Command("start"))
async def cmd_start(message: types.Message) -> None:
    """Handle the /start command."""
    greeting = (
        "👋 Привет! Я Giftbitrage Bot.\n"
        "Отправь команду /scan для поиска выгодных подарков.\n"
        "Ты можешь указать параметры через пробел: минимальная цена, максимальная цена, минимальный % прибыли.\n"
        "Пример: /scan 10 100 5 — искать подарки от 10 до 100 TON с минимальной прибылью 5%."
    )
    await message.answer(greeting)


@router.message(Command("scan"))
async def cmd_scan(message: types.Message, command: CommandObject) -> None:
    """Handle the /scan command with optional parameters."""
    # Defaults
    DEFAULT_MIN_PRICE = 0.0
    DEFAULT_MAX_PRICE = 10_000.0
    DEFAULT_MIN_PROFIT = 5.0
    # Parse arguments
    args = command.args.split() if command.args else []
    try:
        if len(args) >= 1:
            min_price = float(args[0])
        else:
            min_price = DEFAULT_MIN_PRICE
        if len(args) >= 2:
            max_price = float(args[1])
        else:
            max_price = DEFAULT_MAX_PRICE
        if len(args) >= 3:
            min_profit = float(args[2])
        else:
            min_profit = DEFAULT_MIN_PROFIT
    except ValueError:
        await message.answer(
            "Не удалось распознать параметры. Пожалуйста, укажи числа через пробел: "
            "минимальная цена, максимальная цена, минимальный % прибыли."
        )
        return
    # Read auth credentials from environment
    tonnel_auth = os.getenv("TONNEL_AUTH_DATA")
    portals_api_id = os.getenv("PORTALS_API_ID")
    portals_api_hash = os.getenv("PORTALS_API_HASH")
    if not all([tonnel_auth, portals_api_id, portals_api_hash]):
        await message.answer(
            "❗️ Не заданы необходимые переменные окружения: "
            "TONNEL_AUTH_DATA, PORTALS_API_ID, PORTALS_API_HASH. "
            "Установи их в .env файле и перезапусти бота."
        )
        return
    await message.answer(
        f"🔍 Запускаю поиск...\n"
        f"Диапазон цены: {min_price} – {max_price} TON, минимальная прибыль: {min_profit}%"
    )
    try:
        candidates = await scan_and_find(
            tonnel_auth=tonnel_auth,
            portals_api_id=portals_api_id,
            portals_api_hash=portals_api_hash,
            min_price=min_price,
            max_price=max_price,
            min_profit_percent=min_profit,
        )
    except Exception as exc:
        await message.answer(f"Произошла ошибка при сканировании: {exc}")
        return
    response = format_candidates_message(candidates)
    # Telegram imposes a limit of 4096 characters per message.  If the
    # response exceeds this limit, split it into multiple messages.
    MAX_MESSAGE_LENGTH = 4096
    if len(response) <= MAX_MESSAGE_LENGTH:
        await message.answer(response)
    else:
        # Break the response into chunks.  Try to break on newline to
        # maintain readability, but fall back to fixed length if necessary.
        lines = response.split("\n")
        buffer = ""
        for line in lines:
            # +1 for the newline character that was removed by split
            if len(buffer) + len(line) + 1 > MAX_MESSAGE_LENGTH:
                await message.answer(buffer)
                buffer = line
            else:
                buffer = buffer + "\n" + line if buffer else line
        if buffer:
            await message.answer(buffer)


async def main() -> None:
    """Entrypoint for running the bot."""
    load_dotenv()
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN must be set in .env")
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    try:
        await dp.start_polling(bot)
    except TelegramNetworkError as err:
        print(f"Telegram network error: {err}")


if __name__ == "__main__":
    asyncio.run(main())