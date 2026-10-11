"""
refinitiv-data 1.6.2 ile httpx >= 0.28 uyumsuzluğu
===================================================
Kütüphane httpx.Client'a `proxy=<dict>` verir; httpx 0.28 yalnızca None, metin
ya da Proxy nesnesi kabul eder ve oturum açılırken
"'dict' object has no attribute 'url'" hatası verir. Bu modül, kütüphane
dosyalarına dokunmadan, boş sözlüğü None'a çevirir. LSEG'ye bağlanan her
script `rd`'yi kullanmadan önce bunu içe aktarmalıdır.
"""
import httpx
import refinitiv.data._core.session.http_service as _hs


def _get_client(proxies, **kwargs):
    if isinstance(proxies, dict):
        proxies = proxies.get("https://") or proxies.get("http://") or \
                  next(iter(proxies.values()), None)
    return httpx.Client(proxy=proxies, **kwargs)


def _get_client_async(proxies, **kwargs):
    if isinstance(proxies, dict):
        proxies = proxies.get("https://") or proxies.get("http://") or \
                  next(iter(proxies.values()), None)
    return httpx.AsyncClient(proxy=proxies, **kwargs)


_hs.get_httpx_client = _get_client
_hs.get_httpx_client_async = _get_client_async
