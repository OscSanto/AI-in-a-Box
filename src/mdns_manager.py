"""
mDNS advertising and discovery for AIIAB.

Advertises: _aiiab-server._tcp on port 5050
Discovers:  _aiiab-inference._tcp (Android inference app)

Uses the zeroconf library (pip install zeroconf).
Discovery callbacks are thread-bridged into the asyncio event loop.
"""
import asyncio
import socket
import logging
from typing import Callable, Awaitable

from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

log = logging.getLogger(__name__)

ADVERTISE_TYPE = "_aiiab-server._tcp.local."
DISCOVER_TYPE  = "_aiiab-inference._tcp.local."
ADVERTISE_PORT = 5050


def _local_ip() -> str:
    """Best-effort: return the Pi's primary non-loopback IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class MdnsManager:
    """
    Lifecycle:
        mgr = MdnsManager(on_found, on_lost)
        await mgr.start()
        ...
        await mgr.stop()

    on_android_found(host: str, port: int) — called when inference app is found
    on_android_lost()                       — called when it disappears
    """

    def __init__(
        self,
        on_android_found: Callable[[str, int], Awaitable[None]],
        on_android_lost:  Callable[[], Awaitable[None]],
    ):
        self._on_found  = on_android_found
        self._on_lost   = on_android_lost
        self._azc: AsyncZeroconf | None = None
        self._info: ServiceInfo | None  = None
        self._browser: AsyncServiceBrowser | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._azc  = AsyncZeroconf(ip_version=4)

        await self._advertise()
        await self._discover()
        log.info("[mdns] started (advertising %s, discovering %s)",
                 ADVERTISE_TYPE, DISCOVER_TYPE)

    async def stop(self):
        if self._azc:
            try:
                if self._info:
                    await self._azc.async_unregister_service(self._info)
            except Exception:
                pass
            await self._azc.async_close()
            self._azc = None
        log.info("[mdns] stopped")

    # ── Advertising ───────────────────────────────────────────────────────────

    async def _advertise(self):
        ip   = _local_ip()
        desc = {"version": "1"}
        self._info = ServiceInfo(
            type_=ADVERTISE_TYPE,
            name=f"aiiab-server.{ADVERTISE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=ADVERTISE_PORT,
            properties=desc,
            server=f"{socket.gethostname()}.local.",
        )
        try:
            await self._azc.async_register_service(self._info)
            log.info("[mdns] advertising %s on %s:%d", ADVERTISE_TYPE, ip, ADVERTISE_PORT)
        except Exception as e:
            log.warning("[mdns] advertise failed: %s", e)

    # ── Discovery ─────────────────────────────────────────────────────────────

    async def _discover(self):
        self._browser = AsyncServiceBrowser(
            self._azc.zeroconf,
            DISCOVER_TYPE,
            handlers=[self._on_service_state_change],
        )

    def _on_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ):
        if state_change == ServiceStateChange.Added:
            asyncio.run_coroutine_threadsafe(
                self._resolve_and_notify(zeroconf, service_type, name),
                self._loop,
            )
        elif state_change == ServiceStateChange.Removed:
            log.info("[mdns] inference app lost: %s", name)
            asyncio.run_coroutine_threadsafe(self._on_lost(), self._loop)

    async def _resolve_and_notify(self, zeroconf: Zeroconf,
                                   service_type: str, name: str):
        info = ServiceInfo(service_type, name)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: info.request(zeroconf, timeout=3000)
        )
        addresses = info.parsed_scoped_addresses()
        if not addresses:
            log.warning("[mdns] could not resolve %s", name)
            return
        host = addresses[0]
        port = info.port
        log.info("[mdns] inference app found: %s:%d", host, port)
        await self._on_found(host, port)
