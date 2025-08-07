import json
import urllib.parse
from curl_cffi import requests
from tonnelmp.wtf import generate_wtf
from datetime import datetime, timezone, timedelta
from fake_useragent import UserAgent

HEADERS = {
    "authority": "",
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "content-type": "application/json",
    "origin": "https://market.tonnel.network",
    "priority": "u=1, i",
    "referer": "https://market.tonnel.network/",
    "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "",
    }

def tonneltitle(text):
    words = re.findall(r"\w+(?:'\w+)?", text)
    for word in words:
        if len(word) > 0:
            cap = word[0].upper() + word[1:]
            text = text.replace(word, cap, 1)
    return text

def getGifts(
    gift_name: str = None,
    model: str = None,
    backdrop: str = None,
    symbol: str = None,
    gift_num: int = None,
    page: int = 1,
    limit: int = 30,
    sort: str = "price_asc",
    price_range: list | int = 0,
    asset: str = "TON",
    premarket: bool = False,
    telegramMarketplace: bool = False,
    mintable: bool = False,
    bundle: bool = False,
    authData: str = "",
    proxies: dict = None
) -> list:
    
    """
    Retrieve a list of gifts from Tonnel Marketplace

    Args:
        gift_name (str): The name of the gift to filter by.
        model (str): The model of the gift to filter by.
        backdrop (str): The backdrop of the gift to filter by.
        symbol (str): The symbol of the gift to filter by.
        gift_num (int): The gift number to filter by.
        page (int): The page number to retrieve. Default is 1.
        limit (int): The number of gifts to retrieve per page. Default = maximum limit = 30.
        sort (str): The sorting method to apply to the results. Available options: "price_asc", "price_desc", "latest", "mint_time", "rarity", "gift_id_asc", "gift_id_desc"
        price_range (list | int): The price range to filter by. If a list is provided, it should contain two integers: the minimum and maximum price. Default is 0 (no filter).
        asset (str): The asset to filter by. Default is "TON". Available options: "TON", "USDT", "TONNEL"
        premarket (bool): Show only premarket gifts. Default is False
        telegramMarketplace (bool): Show gifts only from Telegram Marketplace. Default is False
        mintable (bool): Show only mintable gifts. Default is False
        bundle (bool): Show only bundle gifts. Default is False
        authData (str): The user auth data required for authorization. Optional.
    Returns:
        list: A list of dict objects with gifts details. 
    """

    URL = "https://gifts2.tonnel.network/api/pageGifts"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    SORTS = {
        "price_asc": "{\"price\":1,\"gift_id\":-1}",
        "price_desc": "{\"price\":-1,\"gift_id\":-1}",
        "latest": "{\"message_post_time\":-1,\"gift_id\":-1}",
        "mint_time": "{\"export_at\":1,\"gift_id\":-1}",
        "rarity": "{\"rarity\":-1,\"gift_id\":-1}",
        "gift_id_asc": "{\"gift_num\":1,\"gift_id\":-1}",
        "gift_id_desc": "{\"gift_num\":-1,\"gift_id\":-1}",
        "model_rarity": "{\"modelRarity\":1,\"gift_id\":-1}",
        "backdrop_rarity": "{\"backdropRarity\":1,\"gift_id\":-1}",
        "symbol_rarity": "{\"symbolRarity\":1,\"gift_id\":-1}"
    }

    try:
        sort_value = SORTS[sort]
    except KeyError:
        raise Exception("tonnelmp: getGifts(): Invalid sort argument. Available sorts: " + str(list(SORTS.keys())))

    filter_dict = {
        "price": {"$exists": True},
        "buyer": {"$exists": False},
        "asset": asset
    }

    if premarket:
        filter_dict["premarket"] = True

    if telegramMarketplace:
        filter_dict["telegramMarketplace"] = True
        filter_dict["export_at"] = {"$exists": False}

    if mintable:
        now_iso = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        if "export_at" in filter_dict:
            filter_dict["$and"] = [
                {"export_at": filter_dict.pop("export_at")},
                {"export_at": {"$lt": now_iso}}
            ]
        else:
            filter_dict["export_at"] = {"$lt": now_iso}

    if bundle:
        filter_dict["gift_id"] = {"$lt": 0}

    if not (premarket or telegramMarketplace or mintable or bundle):
        filter_dict["refunded"] = {"$ne": True}
        filter_dict["export_at"] = {"$exists": True}
    
    if gift_name:
        if gift_name.lower() == "jack-in-the-box":
            filter_dict["gift_name"] = "Jack-in-the-Box"
        else:
            filter_dict["gift_name"] = tonneltitle(gift_name.strip())

    if model:
        if "(" not in model:
            filter_dict["model"] = {"$regex": f"^{tonneltitle(model.strip())} \\("}
        else:
            filter_dict["model"] = f"{tonneltitle(model.strip())}"
            

    if backdrop:
        if "(" not in backdrop:
            filter_dict["backdrop"] = {"$regex": f"^{tonneltitle(backdrop.strip())} \\("}
        else:
            filter_dict["backdrop"] = f"{tonneltitle(backdrop.strip())}"

    if symbol:
        if "(" not in symbol:
            filter_dict["symbol"] = {"$regex": f"^{tonneltitle(symbol.strip())} \\("}
        else:
            filter_dict["symbol"] = f"{tonneltitle(symbol.strip())}"

    if gift_num is not None:
        filter_dict["gift_num"] = str(gift_num)

    payload = {
        "filter": json.dumps(filter_dict),
        "limit": limit,
        "page": page,
        "sort": sort_value,
        "ref": 0,
        "user_auth": authData
    }

    if isinstance(price_range, list) and len(price_range) == 2:
        payload["price_range"] = price_range
    else:
        payload["price_range"] = 0

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: getGifts(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: getGifts(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: getGifts(): Request failed with status code {response.status_code}")

    return response.json()

def myGifts(
    listed: bool = True,
    page: int = 1,
    limit: int = 30,
    authData: str = "",
    proxies: dict | None = None
) -> list:
    
    """
    [Requires authentication]
    Fetches a list of your gifts on the market. The list can be filtered to
    either show only the gifts that are currently listed, or all the gifts
    that are unlisted.

    Args:
        listed: A boolean indicating whether to show only the listed gifts
            or all the non-listed gifts. Default is True.
        page: An integer indicating which page of the results to fetch.
            Default is 1.
        limit: An integer indicating how many results to fetch per page.
            Default is 30.
        authData: The user's auth data required for authorization.

    Returns:
        list: A list of dict objects with gifts details. 
    """
    if not authData:
        raise ValueError("tonnelmp: myGifts(): authData is required")

    URL = "https://gifts2.tonnel.network/api/pageGifts"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    parsed = urllib.parse.parse_qs(authData)
    user_data_json = parsed.get("user", ["{}"])[0]
    try:
        user_data = json.loads(urllib.parse.unquote(user_data_json))
        user_id = user_data["id"]
    except (json.JSONDecodeError, KeyError):
        raise Exception("tonnelmp: myGifts(): Invalid authData format — could not extract user ID")

    if listed:
        sort_value = "{\"message_post_time\":-1,\"gift_id\":-1}"
        filter_dict = {
            "seller": user_id,
            "buyer": {"$exists": False},
            "$or": [
                {"price": {"$exists": True}},
                {"auction_id": {"$exists": True}}
            ]
        }
    else:
        sort_value = "{\"gift_num\":1,\"gift_id\":-1}"
        filter_dict = {
            "seller": user_id,
            "buyer": {"$exists": False},
            "refunded": {"$ne": True},
            "price": {"$exists": False}
        }

    payload = {
        "filter": json.dumps(filter_dict),
        "limit": limit,
        "page": page,
        "sort": sort_value,
        "ref": 0,
        "user_auth": authData
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: myGifts(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: myGifts(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: myGifts(): Request failed with status code {response.status_code}")

    return response.json()

def listForSale(
    gift_id: int,
    price: int | float,
    authData: str,
    proxies: dict | None = None
    ) -> dict:
    """
    [Requires authentication]
    Lists a gift for sale on the marketplace.

    Args:
        gift_id (int): The Tonnel Gift ID of the gift (not gift_num / telegram gift number) to be listed for sale. Can be retrieved from myGifts or getGifts.
        price (int | float): The price at which the gift will be listed.
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: A dictionary containing the response data from the API. Either success or error.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """

    if not authData:
        raise ValueError("tonnelmp: listForSale(): authData is required")

    URL = "https://gifts.coffin.meme/api/listForSale"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    timestamp, wtf = generate_wtf()

    payload = {
        "authData": authData,
        "gift_id": gift_id,
        "price": price,
        "asset": "TON",
        "timestamp": timestamp,
        "wtf": wtf
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: listForSale(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: listForSale(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: listForSale(): Request failed with status code {response.status_code}")

    return response.json()

def cancelSale(
    gift_id: int,
    authData: str,
    proxies: dict | None = None
    ) -> dict:
    
    """
    [Requires authentication]
    Cancels a gift that is currently listed for sale.

    Args:
        gift_id (int): The Tonnel Gift ID of the gift to cancel sale of (not gift_num / telegram gift number). Can be retrieved from myGifts or getGifts.
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: A dictionary containing the response data from the API. Either success or error.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """
    if not authData:
        raise ValueError("tonnelmp: cancelSale(): authData is required")

    URL = "https://gifts.coffin.meme/api/cancelSale"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    timestamp, wtf = generate_wtf()

    payload = {
        "authData": authData,
        "gift_id": gift_id,
        "timestamp": timestamp,
        "wtf": wtf
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: cancelSale(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: cancelSale(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: cancelSale(): Request failed with status code {response.status_code}")

    return response.json()

def saleHistory(
    authData: str,
    page: int = 1,
    limit: int = 50,
    type: str = "ALL",
    gift_name: str = None,
    model: str = None,
    backdrop: str = None,
    sort: str = "latest",
    proxies: dict | None = None
) -> list:
    
    """
    [Requires authentication]
    Retrieves a list of gifts sold on the marketplace.

    Args:
        authData (str): The user's auth data required for authorization.
        page (int): The page number to retrieve. Default is 1.
        limit (int): The number of results to retrieve per page. Default = 50.
        type (str): The type of gifts to filter by. Available options: "ALL", "SALE", "INTERNAL_SALE", "BID"
        gift_name (str): The name of the gift to filter by.
        model (str): The model of the gift to filter by.
        backdrop (str): The backdrop of the gift to filter by.
        sort (str): The sorting method to apply to the results. Default is "latest". Available options: "latest", "price_asc", "price_desc", "gift_id_asc", "gift_id_desc".

    Returns:
        list: A list of dict objects with gifts details.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """
    
    URL = "https://gifts2.tonnel.network/api/saleHistory"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    SORTS = {
        "latest": "{\"timestamp\":-1,\"gift_id\":-1}",
        "price_asc": "{\"price\":1,\"gift_id\":-1}",
        "price_desc": "{\"price\":-1,\"gift_id\":-1}",
        "gift_id_asc": "{\"gift_num\":1,\"gift_id\":-1}",
        "gift_id_desc": "{\"gift_num\":-1,\"gift_id\":-1}"
    }

    try:
        sort_value = SORTS[sort]
    except KeyError:
        raise Exception("tonnelmp: saleHistory(): Invalid sort argument. Available sorts: " + str(list(SORTS.keys())))

    filter_dict = {}

    if gift_name:
        if gift_name.lower() == "jack-in-the-box":
            filter_dict["gift_name"] = "Jack-in-the-Box"
        else:
            filter_dict["gift_name"] = tonneltitle(gift_name.strip())

    if model:
        if "(" not in model:
            filter_dict["model"] = {"$regex": f"^{tonneltitle(model.strip())} \\("}
        else:
            filter_dict["model"] = f"{tonneltitle(model.strip())}"

    if backdrop:
        if "(" not in backdrop:
            filter_dict["backdrop"] = {"$regex": f"^{tonneltitle(backdrop.strip())} \\("}
        else:
            filter_dict["backdrop"] = f"{tonneltitle(backdrop.strip())}"

    payload = {
        "authData": authData,
        "page": page,
        "limit": limit,
        "type": type,
        "filter": filter_dict,
        "sort": json.loads(sort_value)
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: saleHistory(): Request failed with exception {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: saleHistory(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: saleHistory(): equest failed with status code {response.status_code}")

    return response.json()

def getAuctions(
    gift_name: str = None,
    model: str = None,
    backdrop: str = None,
    symbol: str = None,
    gift_num: int = None,
    page: int = 1,
    limit: int = 30,
    sort: str = "ending_soon",
    price_range: list | int = 0,
    asset: str = "TON",
    authData: str = "",
    proxies: dict | None = None
) -> list:
    """
    Retrieves a list of auctions on the marketplace.

    Args:
        gift_name: The name of the gift to filter by.
        model: The model of the gift to filter by.
        backdrop: The backdrop of the gift to filter by.
        symbol: The symbol of the gift to filter by.
        gift_num: The gift number to filter by.
        page: An integer indicating which page of the results to fetch.
            Default is 1.
        limit: An integer indicating how many results to fetch per page.
            Default is 30.
        sort: A string indicating how to sort the results. Default is "ending_soon". Available options: "ending_soon", "latest", "highest_bid", "latest_bid".
        price_range: A list or integer indicating the price range to filter by.
            If a list is provided, it should contain two integers: the minimum
            and maximum price. If an integer is provided, it will be used as the
            minimum price and the maximum price will be set to infinity.
            Default is 0 (no filter).
        asset: A string indicating which asset to filter by. Default is "TON". Available options: "TON", "USDT", "TONNEL".
        authData: The user auth data required for authorization. Optional.

    Returns:
        list: A list of dict objects with auctions details.
    """
    URL = "https://gifts2.tonnel.network/api/pageGifts"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    SORTS = {
        "ending_soon": "{\"auctionEndTime\":1,\"gift_id\":-1}",
        "latest": "{\"auctionStartTime\":-1,\"gift_id\":-1}",
        "highest_bid": "{\"bidHistory.amount\":-1,\"gift_id\":-1}",
        "latest_bid": "{\"bidHistory.timestamp\":-1,\"gift_id\":-1}"
    }

    try:
        sort_value = SORTS[sort]
    except KeyError:
        raise Exception("tonnelmp: getAuctions(): Invalid sort argument. Available sorts: " + str(list(SORTS.keys())))

    filter_dict = {
        "auction_id": {"$exists": True},
        "status": "active",
        "asset": asset
    }
    if gift_name:
        if gift_name.lower() == "jack-in-the-box":
            filter_dict["gift_name"] = "Jack-in-the-Box"
        else:
            filter_dict["gift_name"] = tonneltitle(gift_name.strip())

    if model:
        if "(" not in model:
            filter_dict["model"] = {"$regex": f"^{tonneltitle(model.strip())} \\("}
        else:
            filter_dict["model"] = f"{tonneltitle(model.strip())}"

    if backdrop:
        if "(" not in backdrop:
            filter_dict["backdrop"] = {"$regex": f"^{tonneltitle(backdrop.strip())} \\("}
        else:
            filter_dict["backdrop"] = f"{tonneltitle(backdrop.strip())}"

    if symbol:
        if "(" not in symbol:
            filter_dict["symbol"] = {"$regex": f"^{tonneltitle(symbol.strip())} \\("}
        else:
            filter_dict["symbol"] = f"{tonneltitle(symbol.strip())}"

    if gift_num:
        filter_dict["gift_num"] = str(gift_num)

    payload = {
        "filter": json.dumps(filter_dict),
        "limit": limit,
        "page": page,
        "sort": sort_value,
        "ref": 0,
        "user_auth": authData
    }

    if isinstance(price_range, list) and len(price_range) == 2:
        payload["price_range"] = price_range
    else:
        payload["price_range"] = 0
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: getAuctions(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: getAuctions(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: getAuctions(): Request failed with status code {response.status_code}")

    return response.json()

def iso(hours: int) -> str:
    dt = datetime.now(timezone.utc) + timedelta(hours=hours)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

def createAuction(
    gift_id: int,
    starting_bid: int | float,
    authData: str,
    duration: int = 1,
    proxies: dict | None = None
) -> dict:
    
    """
    [Requires authentication]
    Creates an auction for a gift on the marketplace.

    Args:
        gift_id (int): The Tonnel Gift ID of the gift (not gift_num / telegram gift number) to be auctioned. Can be retrieved from myGifts or getGifts.
        starting_bid (int | float): The starting bid price of the auction.
        authData (str): The user's auth data required for authorization.
        duration (int): The duration of the auction in hours. Default is 1. Available options: 1, 2, 3, 6, 12, 24

    Returns:
        dict: A dictionary containing the response data from the API. Either success or error.

    Raises:
        ValueError: If authData is not provided.
        ValueError: If duration is not one of the allowed durations.
        Exception: If the API request fails, with details about the status code and response text.
    """
    
    ALLOWED_DURATIONS = {1, 2, 3, 6, 12, 24}
    if not authData:
        raise ValueError("authData is required")

    if duration not in ALLOWED_DURATIONS:
        raise ValueError(
            f"tonnelmp: createAuction(): duration_hours must be one of {sorted(ALLOWED_DURATIONS)} "
            f"(got {duration})"
        )

    auction_end_time = iso(duration)

    payload = {
        "authData"     : authData,
        "gift_id"      : gift_id,
        "startingBid"  : starting_bid,
        "auctionEndTime": auction_end_time
    }

    URL = "https://gifts.coffin.meme/api/auction/create"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: createAuction(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: createAuction(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: createAuction(): Request failed with status code {response.status_code}")

    return response.json()

def cancelAuction(
    auction_id: str,
    authData: str,
    proxies: dict | None = None
    ) -> dict:
    """
    [Requires authentication]
    Cancels an auction on Tonnel Marketplace.

    Args:
        auction_id (str): The ID of the auction to cancel. Can be retrieved from getAuctions or myGifts.
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: A dictionary containing the response data from the API. Either success or error.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """
    if not authData:
        raise ValueError("tonnelmp: cancelAuction(): authData is required")
    
    URL = "https://gifts.coffin.meme/api/auction/cancel"
    
    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData,
        "auction_id": auction_id
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: cancelAuction(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: cancelAuction(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: cancelAuction(): Request failed with status code {response.status_code}")

    return response.json()

def buyGift(
    gift_id: int,
    price: int | float,
    authData: str,
    receiver: str = None,
    anonymously: bool = False,
    showPrice: bool = False,
    proxies: dict | None = None
) -> dict:
    """
    [Requires authentication]
    Purchases or gifts a gift from the marketplace.

    Args:
        gift_id (int): The Tonnel Gift ID of the gift to purchase (not gift_num / telegram gift number).
        price (int | float): The price at which the gift is to be purchased.
        authData (str): The user's auth data required for authorization.
        receiver (int, optional): Telegram ID of the gift recipient (if gifting). Tested only with int values, may work with str, may not.
        anonymously (bool, optional): Whether to gift anonymously. Defaults to False.
        showPrice (bool, optional): Whether to show the gift price to the recipient. Defaults to False.

    Returns:
        dict: A dictionary containing the response data from the API. Either success or error.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """

    if not authData:
        raise ValueError("tonnelmp: buyGift(): authData is required")

    URL = f"https://gifts.coffin.meme/api/buyGift/{gift_id}"
    
    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    timestamp, wtf = generate_wtf()

    payload = {
        "authData": authData,
        "asset": "TON",
        "price": float(price),
        "timestamp": timestamp,
        "wtf": wtf
    }

    if receiver:
        payload.update({
            "receiver": receiver,
            "anonymously": anonymously,
            "showPrice": showPrice
        })
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: buyGift(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: buyGift(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: buyGift(): Request failed with status code {response.status_code}")

    return response.json()


def info(
    authData: str,
    proxies: dict | None = None
    ) -> dict:
    """
    [Requires authentication]
    Retrieves information about the user

    Args:
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: A dictionary containing balances, memo etc.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """
    if not authData:
        raise ValueError("tonnelmp: info(): authdata is required")

    URL = "https://gifts2.tonnel.network/api/balance/info"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: info(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: info(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: info(): Request failed with status code {response.status_code}")

    return response.json()

class Gift:
    """
    A class representing a gift on the Tonnel Marketplace.

    Attributes:
        .gift_num : Telegram gift number
        .gift_id : Tonnel Gift ID
        .name : Name of the gift
        .model : Model of the gift
        .backdrop : Backdrop of the gift
        .symbol : Symbol of the gift
        .price : Price of the gift
        .status : Status of the gift
        .asset : Asset of the gift
        .export_at : Export time of the gift
        .customEmojiId : Custom emoji ID
        .auction : Auction details
        .premarket : Premarket status (true / false)
        .bundleData : Bundle details
    """
    def __init__(self, data: dict):
        self._data = data

    @property
    def gift_id(self) -> int:
        return self._data.get("gift_id")

    @property
    def gift_num(self) -> int:
        return self._data.get("gift_num")

    @property
    def name(self) -> str:
        return self._data.get("name")

    @property
    def model(self) -> str:
        return self._data.get("model")

    @property
    def backdrop(self) -> str:
        return self._data.get("backdrop")

    @property
    def symbol(self) -> str:
        return self._data.get("symbol")

    @property
    def price(self) -> float | None:
        return self._data.get("price")

    @property
    def status(self) -> str:
        return self._data.get("status")

    @property
    def asset(self) -> str:
        return self._data.get("asset")

    @property
    def export_at(self) -> str:
        return self._data.get("export_at")

    @property
    def premarket(self) -> bool:
        return self._data.get("premarket")
    
    @property
    def bundleData(self) -> dict:
        return self._data.get("bundleData")

    def to_dict(self) -> dict:
        """Return the raw data as a dictionary."""
        return self._data.copy()

def withdraw(
    wallet: str,
    authData: str,
    amount: int | float,
    asset: str = "TON",
    proxies: dict | None = None
) -> dict:
    """
    [Requires authentication]
    Withdraws a specified amount of an asset to a given TON wallet.

    Args:
        wallet (str): TON wallet address to withdraw to.
        authData (str): Authentication string for the user.
        amount (int | float): >= 0.5. Amount to withdraw (fee of 0.001 will be subtracted internally).
        asset (str): Asset to withdraw. Options: "TON", "USDT", "TONNEL". Defaults to "TON".

    Returns:
        dict: Response from the withdrawal API.

    Raises:
        ValueError: If any required parameters are missing or invalid.
        Exception: If the API request fails.
    """

    if not wallet or not authData or not amount:
        raise ValueError("tonnelmp: withdraw(): wallet, authData, and amount are required fields")

    if asset not in {"TON", "USDT", "TONNEL"}:
        raise ValueError("tonnelmp: withdraw(): Invalid asset type. Must be 'TON', 'USDT', or 'TONNEL'.")

    URL = "https://gifts.coffin.meme/api/balance/withdraw"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "wallet": wallet,
        "authData": authData,
        "amount": float(amount),
        "asset": asset
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=15, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: withdraw(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: withdraw(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: withdraw(): Request failed with status code {response.status_code}")

    return response.json()

def returnGift(gift_id: int, authData: str, proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Returns a purchased gift back to the Tonnel Marketplace.

    Args:
        gift_id (int): Tonnel Gift ID to return.
        authData (str): The user's authentication data.

    Returns:
        dict: The API's response data indicating success or failure.

    Raises:
        ValueError: If required arguments are missing.
        Exception: If the API request fails with a non-200 status.
    """
    if not gift_id or not authData:
        raise ValueError("tonnelmp: returnGift(): Both gift_id and authData are required.")

    URL = "https://gifts.coffin.meme/api/returnGift"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "gift_id": gift_id,
        "authData": authData
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: returnGift(): Request failed with error: {e}")

    if response.status_code == 429:
        raise Exception(f"tonnelmp: returnGift(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: returnGift(): Request failed with status code {response.status_code}")

    return response.json()

def placeBid(auction_id: str, amount: int | float, authData: str, asset: str = "TON", proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Places a bid on an auctioned gift in the Tonnel Marketplace.

    Args:
        auction_id (str): The ID of the auction.
        amount (int | float): The bid amount.
        asset (str): The asset used for the bid ("TON", "USDT", "TONNEL"). Defaults to "TON".
        authData (str): The user's authentication data.

    Returns:
        dict: The response from the API indicating success or error.

    Raises:
        ValueError: If required arguments are missing.
        Exception: If the API request fails or returns a non-200 status.
    """
    if not auction_id or not amount or not authData:
        raise ValueError("tonnelmp: placeBid(): All arguments (auction_id, amount, authData) are required.")

    URL = "https://gifts.coffin.meme/api/auction/bid"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData,
        "auction_id": auction_id,
        "amount": float(amount),
        "asset": asset
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: placeBid(): Request failed with error: {e}")

    if response.status_code == 429:
        raise Exception(f"tonnelmp: placeBid(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: placeBid(): Request failed with status code {response.status_code}")

    return response.json()

def switchTransfer(authData: str, transferGift: bool, proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Toggles internal gift transfer mode in the Tonnel Marketplace.

    Args:
        authData (str): The user's authentication data.
        transferGift (bool): Set to True to enable internal transfer mode, False to disable.

    Returns:
        dict: The response from the API indicating success or failure.

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails.
    """
    if not authData:
        raise ValueError("tonnelmp: switchTransfer(): authData is required.")

    URL = "https://gifts.coffin.meme/api/user/switchTransfer"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "transferGift": transferGift,
        "authData": authData
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: switchTransfer(): Request failed with error: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: switchTransfer(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: switchTransfer(): Request failed with status code {response.status_code}")

    return response.json()

def mintGift(authData: str, wallet: str, gift_id: int, proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Initiates the minting process of a gift to the specified wallet.

    Args:
        authData (str): The user's authentication data.
        wallet (str): The recipient's TON wallet address.
        gift_id (int): The sale ID of the gift to be minted.

    Returns:
        dict: The response from the API indicating success or failure.

    Raises:
        ValueError: If any required argument is missing.
        Exception: If the API request fails.
    """
    if not authData:
        raise ValueError("tonnelmp: mintGift(): authData is required.")
    if not wallet:
        raise ValueError("tonnelmp: mintGift(): wallet is required.")
    if not gift_id:
        raise ValueError("tonnelmp: mintGift(): gift_id is required.")

    URL = "https://gifts.coffin.meme/api/mint/start"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData,
        "wallet": wallet,
        "gift_id": gift_id
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=15, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: mintGift(): Request failed with error: {e}")
    
    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: mintGift(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: mintGift(): Request failed with status code {response.status_code}")

    return response.json()

def unlockListing(authData: str, gift_id: int, proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Unlocks a listing on the Tonnel Marketplace, allowing it to be relisted or managed.

    Args:
        authData (str): The user's authentication token.
        gift_id (int): The sale ID of the gift/listing to unlock.

    Returns:
        dict: API response indicating success or failure.

    Raises:
        ValueError: If required fields are missing.
        Exception: If the API request fails.
    """
    if not authData:
        raise ValueError("tonnelmp: unlockListing(): authData is required.")
    if not gift_id:
        raise ValueError("tonnelmp: unlockListing(): sale_id is required.")

    URL = "https://gifts.coffin.meme/api/unlock"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData,
        "sale_id": gift_id
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: unlockListing(): Request failed: {e}")
    
    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: unlockListing(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: unlockListing(): Request failed with status code {response.status_code}")

    return response.json()

def giveawayInfo(giveaway_id: str, authData: str, proxies: dict | None = None) -> dict:
    """
    [Requires authentication]
    Retrieves information about a specific giveaway from the Tonnel Marketplace.

    Args:
        giveaway_id (str): The ID of the giveaway to retrieve info about.
        authData (str): The user's authentication token.

    Returns:
        dict: API response containing giveaway information.

    Raises:
        ValueError: If required fields are missing.
        Exception: If the API request fails.
    """
    if not authData:
        raise ValueError("tonnelmp: giveawayInfo(): authData is required.")
    if not giveaway_id:
        raise ValueError("tonnelmp: giveawayInfo(): giveaway_id is required.")

    URL = "https://gifts2.tonnel.network/api/giveaway/info"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    payload = {
        "giveAwayId": giveaway_id,
        "authData": authData
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: giveawayInfo(): Request failed: {e}")
    
    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: giveawayInfo(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: giveawayInfo(): getGiveawayInfo failed {response.status_code}: {response.text}")

    return response.json()

def joinGiveaway(
    giveaway_id: str,
    authData: str,
    ticketCount: int | None = None,
    proxies: dict | None = None
) -> dict:
    """
    Joins a giveaway on the Tonnel Marketplace.

    Args:
        giveaway_id (str): The ID of the giveaway.
        authData (str): Authentication data string.
        ticketCount (int, optional): Number of tickets to use (if the giveaway is paid).

    Returns:
        dict: API response.
    """
    if not authData:
        raise ValueError("tonnelmp: joinGiveaway(): authData is required")

    timestamp, wtf = generate_wtf()

    payload = {
        "authData": authData,
        "giveAwayId": giveaway_id,
        "timestamp": timestamp,
        "wtf": wtf,
    }

    if ticketCount is not None:
        payload["ticketCount"] = ticketCount

    URL = "https://gifts.coffin.meme/api/giveaway/join"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts.coffin.meme"
    HEADERS["user-agent"] = user_agent

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: joinGiveaway(): Request failed: {e}")

    if response.status_code != 200:
        raise Exception(f"tonnelmp: joinGiveaway(): Request failed {response.status_code}: {response.text}")

    return response.json()

def filterStats(authData: str, proxies: dict = None) -> dict:
    """
    Retrieves filter stats from tonnel mp.

    Args:
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: Ungrouped dict with all gifts and models with rarities + floors and stuff like that

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails.
    """
    if not authData:
        raise ValueError("tonnelmp: filterStats(): authData is required")

    URL = "https://gifts2.tonnel.network/api/filterStats"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData
    }

    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: filterStats(): Request failed: {e}")

    if response.status_code != 200:
        raise Exception(f"tonnelmp: filterStats(): Request failed {response.status_code}: {response.text}")

    return response.json()

import re

def filterStatsPretty(authData: str, proxies: dict = None) -> dict:
    """
    Prettier version of filterStats with lowercase keys + summary renamed to 'data'.
    Output format:
    {
        "toy bear": {
            "data": {
                "floorPrice": 14.84,
                "howMany": 2400
            },
            "wizard": {
                "floorPrice": 19,
                "howMany": 36,
                "rarity": 1.5
            }
        }
    }
    """

    if not authData:
        raise ValueError("tonnelmp: filterStatsPretty(): authData is required")

    URL = "https://gifts3.tonnel.network/api/filterStats"
    
    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts3.tonnel.network"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: filterStatsPretty(): Request failed: {e}")

    if response.status_code != 200:
        raise Exception(f"tonnelmp: filterStatsPretty(): Request failed {response.status_code}: {response.text}")

    r = response.json()

    if r.get("status") != "success":
        raise Exception("tonnelmp: filterStatsPretty(): api error: " + r.get("message", "unknown error"))

    rawdata = r.get("data", {})
    data = {}

    for key, value in rawdata.items():
        try:
            gift_name, model = key.split("_", 1)
        except ValueError:
            gift_name = key
            model = "Unknown"

        match = re.match(r"^(.*?)\s*\(([\d.]+)%\)$", model)
        if match:
            model_name, rarity_str = match.groups()
            rarity = float(rarity_str)
        else:
            model_name = model.strip()
            rarity = None

        gift_key = gift_name.strip().lower()
        model_key = model_name.strip().lower()

        floor = value.get("floorPrice")
        how_many = value.get("howMany", 0)

        if gift_key not in data:
            data[gift_key] = {
                "data": {
                    "floorPrice": floor,
                    "howMany": how_many
                }
            }
        else:
            currentFloor = data[gift_key]["data"].get("floorPrice")
            if currentFloor is None or (floor is not None and floor < currentFloor):
                data[gift_key]["data"]["floorPrice"] = floor

            data[gift_key]["data"]["howMany"] += how_many

        data[gift_key][model_key] = {
            "floorPrice": floor,
            "howMany": how_many,
            "rarity": rarity
        }

    return {
        "status": "success",
        "data": data
    }

def giftData(
        gift_id: int | str,
        authData: str,
        proxies: dict | None = None
        ) -> dict:
    """
    [Requires authentication]
    Retrieves data of the gift by gift_id from tonnel

    Args:
        gift_id (int | str): Tonnel gift_id of the gift.
        authData (str): The user's auth data required for authorization.

    Returns:
        dict: A dict with gift data

    Raises:
        ValueError: If authData is not provided.
        Exception: If the API request fails, with details about the status code and response text.
    """

    if not authData:
        raise ValueError("tonnelmp: giftData(): authData is required")

    URL = f"https://gifts2.tonnel.network/api/giftData/{gift_id}"

    ua = UserAgent()
    user_agent = ua.random

    HEADERS["authority"] = "gifts2.tonnel.network"
    HEADERS["user-agent"] = user_agent

    payload = {
        "authData": authData,
        "ref": ""
    }
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, impersonate="chrome110", timeout=10, proxies=proxies)
    except Exception as e:
        raise Exception(f"tonnelmp: giftData(): Request failed: {e}")

    if response.status_code in [403, 429]:
        raise Exception(f"tonnelmp: giftData(): Request failed with status code {response.status_code} (Likely CloudFlare)")
    elif response.status_code != 200:
        raise Exception(f"tonnelmp: giftData(): Request failed with status code {response.status_code}")
