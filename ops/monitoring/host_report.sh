#!/usr/bin/env bash
# Host metrics collector — runs every minute on Mac (launchd) + Linux (systemd timer).
# Posts CPU/RAM/disk/net/temperature/GPU snapshot to the wheretolive admin API.
#
# Required env:
#   WTL_ADMIN_KEY   — the site admin key (from the environment)
#   WTL_API         — admin API base on the production host (default http://localhost:3000)
# Optional:
#   WTL_HOST_ID     — stable host identifier (default: hostname)

set -u

API="${WTL_API:-http://localhost:3000}"
ENDPOINT="$API/api/admin/host-metrics"
HOST_ID="${WTL_HOST_ID:-$(hostname -s 2>/dev/null || hostname)}"
HOSTNAME_FULL="$(hostname 2>/dev/null || echo unknown)"
OS_KIND="$(uname -s)"

# ──────────────────────────────────────────────────────────────────────
# OS-agnostic JSON builders
# ──────────────────────────────────────────────────────────────────────

cpu_load_json() {
  # /proc/loadavg works on Linux; sysctl works on macOS
  if [ -r /proc/loadavg ]; then
    read -r l1 l5 l15 _ < /proc/loadavg
  else
    local raw
    raw=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}')
    l1=$(echo "$raw" | awk '{print $1}')
    l5=$(echo "$raw" | awk '{print $2}')
    l15=$(echo "$raw" | awk '{print $3}')
  fi
  local ncpu
  if [ "$OS_KIND" = "Darwin" ]; then
    ncpu=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)
  else
    ncpu=$(nproc 2>/dev/null || echo 1)
  fi
  printf '{"load1":%s,"load5":%s,"load15":%s,"ncpu":%s}' "$l1" "$l5" "$l15" "$ncpu"
}

mem_json() {
  if [ "$OS_KIND" = "Darwin" ]; then
    # macOS vm_stat — page size 16384 on Apple Silicon
    local ps total free active inactive wired compressed
    ps=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
    total=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    local vm
    vm=$(vm_stat 2>/dev/null)
    free=$(echo "$vm"      | awk '/Pages free/         {gsub(/\./,""); print $3}')
    active=$(echo "$vm"    | awk '/Pages active/       {gsub(/\./,""); print $3}')
    inactive=$(echo "$vm"  | awk '/Pages inactive/     {gsub(/\./,""); print $3}')
    wired=$(echo "$vm"     | awk '/Pages wired down/   {gsub(/\./,""); print $4}')
    compressed=$(echo "$vm"| awk '/Pages occupied by compressor/ {gsub(/\./,""); print $5}')
    # All in bytes.
    free=$(( free * ps ))
    active=$(( active * ps ))
    inactive=$(( inactive * ps ))
    wired=$(( wired * ps ))
    compressed=$(( compressed * ps ))
    local used=$(( wired + compressed + active ))
    printf '{"total":%s,"used":%s,"free":%s,"active":%s,"inactive":%s,"wired":%s,"compressed":%s}' \
      "$total" "$used" "$free" "$active" "$inactive" "$wired" "$compressed"
  else
    # Linux /proc/meminfo
    local total avail free buf cache swap_total swap_free
    total=$(awk '/MemTotal:/ {print $2 * 1024}' /proc/meminfo)
    avail=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
    free=$(awk '/MemFree:/ {print $2 * 1024}' /proc/meminfo)
    buf=$(awk '/Buffers:/ {print $2 * 1024}' /proc/meminfo)
    cache=$(awk '/^Cached:/ {print $2 * 1024}' /proc/meminfo)
    swap_total=$(awk '/SwapTotal:/ {print $2 * 1024}' /proc/meminfo)
    swap_free=$(awk '/SwapFree:/ {print $2 * 1024}' /proc/meminfo)
    local used=$(( total - avail ))
    local swap_used=$(( swap_total - swap_free ))
    printf '{"total":%s,"used":%s,"free":%s,"available":%s,"buffers":%s,"cached":%s,"swap_total":%s,"swap_used":%s}' \
      "$total" "$used" "$free" "$avail" "$buf" "$cache" "$swap_total" "$swap_used"
  fi
}

disks_json() {
  # df --output is Linux-only; fall back to portable parsing
  local first=1
  printf '['
  if [ "$OS_KIND" = "Darwin" ]; then
    # macOS APFS: every mount inside a container (/, /System/Volumes/Data,
    # /System/Volumes/Preboot, /System/Volumes/VM, ...) shares the same
    # `Available` space — they only differ in `Used`. Per-volume rows
    # under-report total disk usage (Data's 60% misses the 11 GB SSV +
    # Preboot + VM swap).
    #
    # Collapse to one row per physical disk (e.g. disk3). For each:
    #   total = container total (any mount's size)
    #   free  = container Available
    #   used  = total - free  (real physical use, not just one volume)
    # This matches what "Disk Utility → Capacity" would show for the SSD.
    # No associative arrays — macOS ships bash 3.2. Track seen disk
    # roots in a single space-separated string and grep against it.
    local _seen=" "
    local _root_disk
    _root_disk=$(df / 2>/dev/null | awk 'NR==2 {print $1}' | sed -E 's@^/dev/(disk[0-9]+).*@\1@')
    while IFS= read -r line; do
      local fs size avail mount disk_root used label
      fs=$(echo "$line"   | awk '{print $1}')
      size=$(echo "$line" | awk "{print \$2 * 1024}")
      avail=$(echo "$line" | awk "{print \$4 * 1024}")
      # Extract device root: "/dev/disk3s5" → "disk3"
      disk_root=$(echo "$fs" | sed -E 's@^/dev/(disk[0-9]+).*@\1@')
      [ -z "$disk_root" ] && continue
      case "$_seen" in
        *" $disk_root "*) continue ;;
      esac
      _seen="$_seen$disk_root "
      # Skip tiny system-reserve partitions (<10 GB) — EFI / recovery / etc.
      if [ "$size" -lt $((10 * 1024 * 1024 * 1024)) ]; then continue; fi
      used=$(( size - avail ))
      if [ "$first" -eq 1 ]; then first=0; else printf ","; fi
      label="$disk_root"
      [ "$disk_root" = "$_root_disk" ] && label="Internal SSD"
      printf "{\"mount\":\"%s\",\"fs\":\"/dev/%s\",\"total\":%s,\"used\":%s,\"free\":%s}" \
        "$label" "$disk_root" "$size" "$used" "$avail"
    done < <(df -k -T apfs,hfs,exfat 2>/dev/null | tail -n +2)
  else
    # 9p = WSL's host-Windows bridge (/mnt/c, /mnt/d, /usr/lib/wsl/drivers).
    # Skip — not the WSL container's own storage AND `fs` like "C:\" contains
    # an unescaped backslash that breaks the assembled JSON.
    # rootfs = WSL2 initramfs scratch, irrelevant for disk monitoring.
    df -B1 --output=source,fstype,size,used,avail,target \
        -x tmpfs -x devtmpfs -x squashfs -x overlay -x efivarfs -x autofs \
        -x cgroup2 -x pstore -x bpf -x tracefs -x debugfs -x ramfs \
        -x 9p -x rootfs 2>/dev/null \
      | tail -n +2 \
      | while IFS= read -r line; do
        local fs ftype size used avail mount
        fs=$(echo "$line"    | awk '{print $1}')
        ftype=$(echo "$line" | awk '{print $2}')
        size=$(echo "$line"  | awk '{print $3}')
        used=$(echo "$line"  | awk '{print $4}')
        avail=$(echo "$line" | awk '{print $5}')
        mount=$(echo "$line" | awk '{for (i=6; i<=NF; i++) printf "%s%s", $i, (i==NF?"":" ")}')
        case "$mount" in
          /boot/efi|/snap*|/run*|/var/snap*|/sys/*|/proc/*) continue ;;
        esac
        if [ "$size" = "0" ]; then continue; fi
        # Defense in depth: escape backslash + double-quote in fs/mount so
        # any edge-case path doesn't blow up the JSON.
        local fs_esc mount_esc
        fs_esc=${fs//\\/\\\\}; fs_esc=${fs_esc//\"/\\\"}
        mount_esc=${mount//\\/\\\\}; mount_esc=${mount_esc//\"/\\\"}
        if [ "$first" -eq 1 ]; then first=0; else printf ','; fi
        printf '{"mount":"%s","fs":"%s","fstype":"%s","total":%s,"used":%s,"free":%s}' \
          "$mount_esc" "$fs_esc" "$ftype" "$size" "$used" "$avail"
      done
  fi
  printf ']'
}

temperature_json() {
  # Linux: sensors output (if installed); macOS: no easy way without sudo
  if [ "$OS_KIND" = "Linux" ] && command -v sensors >/dev/null 2>&1; then
    local pkg
    pkg=$(sensors 2>/dev/null | awk '/Package id 0:/ {gsub(/[+°C]/,""); print $4; exit}')
    if [ -n "$pkg" ]; then
      printf '{"cpu_pkg_c":%s}' "$pkg"
      return
    fi
  fi
  printf 'null'
}

gpu_json() {
  # NVIDIA via nvidia-smi. On WSL2 the binary lives at /usr/lib/wsl/lib/
  # but isn't on PATH — fall back to that path if `command -v` misses.
  local nvsmi=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvsmi="nvidia-smi"
  elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    nvsmi="/usr/lib/wsl/lib/nvidia-smi"
  fi
  if [ -n "$nvsmi" ]; then
    local out
    out=$("$nvsmi" --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null \
          | head -1)
    if [ -n "$out" ]; then
      local name util_pct mem_used mem_total temp_c power_w
      name=$(echo "$out"      | awk -F', ' '{print $1}')
      util_pct=$(echo "$out"  | awk -F', ' '{print $2}')
      mem_used=$(echo "$out"  | awk -F', ' '{print $3}')
      mem_total=$(echo "$out" | awk -F', ' '{print $4}')
      temp_c=$(echo "$out"    | awk -F', ' '{print $5}')
      power_w=$(echo "$out"   | awk -F', ' '{print $6}')
      # power.draw is "[N/A]" on some headless / passthrough GPUs (incl. WSL).
      # Replace any non-numeric with null so JSON.parse on the admin side doesn't choke.
      [[ "$util_pct"  =~ ^[0-9.]+$ ]] || util_pct=null
      [[ "$mem_used"  =~ ^[0-9.]+$ ]] || mem_used=null
      [[ "$mem_total" =~ ^[0-9.]+$ ]] || mem_total=null
      [[ "$temp_c"    =~ ^[0-9.]+$ ]] || temp_c=null
      [[ "$power_w"   =~ ^[0-9.]+$ ]] || power_w=null
      printf '{"vendor":"nvidia","name":"%s","util_pct":%s,"mem_used_mb":%s,"mem_total_mb":%s,"temp_c":%s,"power_w":%s}' \
        "$name" "$util_pct" "$mem_used" "$mem_total" "$temp_c" "$power_w"
      return
    fi
  fi
  printf 'null'
}

diskio_json() {
  # 1s iostat sample. First sample is since-boot; second is the delta.
  # Costs ~1s of wall-clock — fine at 1-min cadence.
  if [ "$OS_KIND" = "Darwin" ]; then
    if command -v iostat >/dev/null 2>&1; then
      local last
      last=$(iostat -d -w 1 -c 2 disk0 2>/dev/null | tail -1)
      if [ -n "$last" ]; then
        local kbt tps mbs
        kbt=$(echo "$last" | awk '{print $1}')
        tps=$(echo "$last" | awk '{print $2}')
        mbs=$(echo "$last" | awk '{print $3}')
        printf '[{"device":"disk0","tps":%s,"mb_s":%s,"kb_per_t":%s}]' \
          "${tps:-0}" "${mbs:-0}" "${kbt:-0}"
        return
      fi
    fi
    printf '[]'
    return
  fi
  if command -v iostat >/dev/null 2>&1; then
    local block
    block=$(iostat -dxk 1 2 2>/dev/null | awk '
      /^Device/ {section++; next}
      section == 2 && $1 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+|hd[a-z]|vd[a-z])$/ {
        printf "{\"device\":\"%s\",\"r_per_s\":%s,\"w_per_s\":%s,\"r_kb_s\":%s,\"w_kb_s\":%s,\"util_pct\":%s}\n",
               $1, $2, $8, $3, $9, $NF
      }')
    if [ -n "$block" ]; then
      printf '['
      local first=1
      while IFS= read -r line; do
        [ "$first" -eq 1 ] && first=0 || printf ','
        printf '%s' "$line"
      done <<< "$block"
      printf ']'
      return
    fi
  fi
  # /proc/diskstats fallback — for hosts without sysstat (WSL, minimal Linux).
  # Sample twice with 1s gap; convert sector deltas (512B each) to kB/s.
  # Fields: major minor name r r_m r_sect r_ms w w_m w_sect w_ms ... (Linux 5.x+)
  if [ -r /proc/diskstats ]; then
    local snap1 snap2
    snap1=$(awk '$3 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+|hd[a-z]|vd[a-z])$/ {print $3, $6, $10}' /proc/diskstats)
    sleep 1
    snap2=$(awk '$3 ~ /^(sd[a-z]|nvme[0-9]+n[0-9]+|hd[a-z]|vd[a-z])$/ {print $3, $6, $10}' /proc/diskstats)
    if [ -n "$snap1" ] && [ -n "$snap2" ]; then
      printf '['
      local first=1
      while IFS=' ' read -r dev r1 w1; do
        local r2 w2 r_kb w_kb
        r2=$(echo "$snap2" | awk -v d="$dev" '$1==d {print $2}')
        w2=$(echo "$snap2" | awk -v d="$dev" '$1==d {print $3}')
        [ -z "$r2" ] && continue
        # sectors are 512 B → KB = sectors/2
        r_kb=$(( (r2 - r1) / 2 ))
        w_kb=$(( (w2 - w1) / 2 ))
        [ "$first" -eq 1 ] && first=0 || printf ','
        printf '{"device":"%s","r_kb_s":%s,"w_kb_s":%s}' "$dev" "$r_kb" "$w_kb"
      done <<< "$snap1"
      printf ']'
      return
    fi
  fi
  printf '[]'
}

uptime_json() {
  local up=0
  if [ -r /proc/uptime ]; then
    up=$(awk '{print int($1)}' /proc/uptime)
  else
    # macOS: kern.boottime → "{ sec = 1234567890, usec = 0 } Wed ..."
    # Field 4 of awk = "1234567890," → tr removes trailing comma.
    local bt now
    bt=$(sysctl -n kern.boottime 2>/dev/null | awk '{print $4}' | tr -d ',')
    now=$(date +%s)
    if [ -n "$bt" ] && [ "$bt" -gt 0 ] 2>/dev/null; then
      up=$(( now - bt ))
    fi
  fi
  printf '{"seconds":%s}' "$up"
}

# ──────────────────────────────────────────────────────────────────────
# Assemble + POST
# ──────────────────────────────────────────────────────────────────────

METRICS=$(printf '{"cpu":%s,"mem":%s,"disks":%s,"diskio":%s,"temp":%s,"gpu":%s,"uptime":%s}' \
  "$(cpu_load_json)" \
  "$(mem_json)" \
  "$(disks_json)" \
  "$(diskio_json)" \
  "$(temperature_json)" \
  "$(gpu_json)" \
  "$(uptime_json)")

PAYLOAD=$(printf '{"host_id":"%s","hostname":"%s","os":"%s","metrics":%s}' \
  "$HOST_ID" "$HOSTNAME_FULL" "$OS_KIND" "$METRICS")

if [ -z "${WTL_ADMIN_KEY:-}" ]; then
  echo "[host_report] missing WTL_ADMIN_KEY env" >&2
  exit 1
fi

# Silent on success; print error on failure
RESP=$(curl -sS --max-time 10 \
  -H "Content-Type: application/json" \
  -H "x-admin-key: $WTL_ADMIN_KEY" \
  -X POST "$ENDPOINT" \
  -d "$PAYLOAD" 2>&1)
RC=$?
if [ "$RC" -ne 0 ] || ! echo "$RESP" | grep -q '"ok":true'; then
  echo "[host_report] POST failed (rc=$RC): $RESP" >&2
  exit 1
fi
