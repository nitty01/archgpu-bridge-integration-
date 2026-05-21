#!/usr/bin/env bash
# Pull the latest image tag and recreate the Open WebUI container with the same
# port mappings, volume mounts, extra_hosts, and environment variables.
#
# Usage:
#   OPENWEBUI_NAME=openwebui OPENWEBUI_IMAGE=ghcr.io/open-webui/open-webui:main-slim \
#     ./scripts/upgrade-openwebui.sh
#
# Requires: docker, python3. Stops the running container briefly (chats in RAM are lost).

set -euo pipefail

NAME="${OPENWEBUI_NAME:-openwebui}"
IMAGE="${OPENWEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main-slim}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "No container named $NAME" >&2
  exit 1
fi

echo "==> Pulling $IMAGE"
docker pull "$IMAGE"

INSPECT_JSON=$(docker inspect "$NAME")
export INSPECT_JSON IMAGE NAME

echo "==> Stopping and removing old container (data volumes/binds are kept on disk)"
docker stop "$NAME"
docker rm "$NAME"

# shellcheck disable=SC2016
python3 <<'PY'
import json, os
data = json.loads(os.environ["INSPECT_JSON"])[0]
c = data["Config"]
h = data["HostConfig"]
name = os.environ["NAME"]
image = os.environ["IMAGE"]
cmd = ["docker", "run", "-d", "--name", name, "--label", "archgpu-upgraded=1"]
rp = h.get("RestartPolicy") or {}
rname, rmax = rp.get("Name", "unless-stopped"), rp.get("MaximumRetryCount", 0)
if rname and rname != "no":
    cmd.extend(["--restart", rname])
nm = h.get("NetworkMode") or ""
if nm == "host":
    cmd.extend(["--network", "host"])
# Port bindings: {'8080/tcp': [{'HostIp': '', 'HostPort': '3000'}]} -> -p 3000:8080
if nm != "host":
    for cport, maps in (h.get("PortBindings") or {}).items():
        in_port, _proto = (cport.split("/") + ["tcp"])[:2]
        for m in maps or []:
            hip, hp = m.get("HostIp") or "", m.get("HostPort") or ""
            if not hp:
                continue
            if hip in ("0.0.0.0", ""):
                cmd.extend(["-p", f"{hp}:{in_port}"])
            else:
                cmd.extend(["-p", f"{hip}:{hp}:{in_port}"])
# Mounts (use Binds from HostConfig; bind mounts in Mounts are redundant)
if h.get("Binds"):
    for b in h["Binds"]:
        cmd.extend(["-v", b])
else:
    for m in data.get("Mounts") or []:
        if m.get("Type") == "bind" and m.get("Source") and m.get("Destination"):
            src, dst = m["Source"], m["Destination"]
            b = f"{src}:{dst}"
            if m.get("RW") is False:
                b += ":ro"
            cmd.extend(["-v", b])
# Extra hosts
for eh in h.get("ExtraHosts") or []:
    cmd.extend(["--add-host", eh])
# Every env the old container had (image defaults + yours)
for e in c.get("Env") or []:
    if e.startswith("com.docker."):
        continue
    cmd.extend(["-e", e])
cmd.append(image)
print("==> Recreating container...")
os.execvp(cmd[0], cmd)
PY
