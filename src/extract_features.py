import re
from urllib.parse import urlparse, parse_qs

SUSPICIOUS_WORDS = {
    "secure","account","update","login","verify","bank","confirm","free","bonus","gift"
}
SHORTENER_DOMAINS = {"bit.ly","t.co","goo.gl","tinyurl.com","is.gd","cutt.ly","ow.ly"}

def extract_url_features(url: str) -> dict:
    """
    Turn a URL into a fixed set of numeric/boolean features.
    The order of keys is stable, which keeps training and prediction aligned.
    """
    url = url.strip()

    # Parse once
    p = urlparse(url)
    host = (p.hostname or "").lower()
    path = p.path or ""
    query = p.query or ""
    scheme = (p.scheme or "").lower()

    has_ip = bool(re.fullmatch(
        r"(?:\d{1,3}\.){3}\d{1,3}", host
    ))

    feats = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "num_subdomains": max(0, len([s for s in host.split(".") if s]) - 2),
        "num_params": len(parse_qs(query)),
        "has_at": int("@" in url),
        "has_ip": int(has_ip),
        "is_https": int(scheme == "https"),
        "is_shortened": int(any(host.endswith(d) for d in SHORTENER_DOMAINS)),
        "has_suspicious_words": int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
    }
    return feats
