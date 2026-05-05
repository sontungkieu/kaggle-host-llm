from __future__ import annotations

import json
import socket
from pathlib import Path
from urllib.request import urlopen


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def get_public_ip() -> str:
    with urlopen("https://api.ipify.org", timeout=5) as response:
        return response.read().decode("utf-8").strip()


def check_tcp(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    env = load_env(Path(".secrets/.env"))
    public_ip = get_public_ip()
    gateway_health_url = env.get("GATEWAY_HEALTH_URL", "http://127.0.0.1:8000/health")
    gateway_ws_url = env.get("GATEWAY_WS_URL", "")
    local_health = {}
    try:
        with urlopen(gateway_health_url, timeout=5) as response:
            local_health = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        local_health = {"error": str(exc)}

    print(f"public_ip={public_ip}")
    print(f"gateway_ws_url={gateway_ws_url}")
    print(f"gateway_health_url={gateway_health_url}")
    print(f"local_health={local_health}")
    print(f"public_tcp_8000_from_here={check_tcp(public_ip, 8000)}")
    print()
    print("If local_health is ok but external devices cannot reach http://PUBLIC_IP:8000/health:")
    print("- verify router port forward: WAN TCP 8000 -> Windows LAN IP TCP 8000")
    print("- verify router WAN IP equals public_ip above")
    print("- if router WAN IP is private/CGNAT, direct public IP hosting will not work")


if __name__ == "__main__":
    main()

