#!/bin/bash
# ─────────────────────────────────────────────────────────────
# swappiness.sh — tune swap behavior for AIIAB on Raspberry Pi
#
# Usage:
#   ./swappiness.sh apply   — set low swappiness + clear swap
#   ./swappiness.sh revert  — restore default swappiness
# ─────────────────────────────────────────────────────────────

SYSCTL_CONF="/etc/sysctl.conf"
LOW_SWAPPINESS=10
DEFAULT_SWAPPINESS=60

case "$1" in
  apply)
    echo "→ Setting swappiness to $LOW_SWAPPINESS..."
    sudo sysctl vm.swappiness=$LOW_SWAPPINESS

    echo "→ Making permanent in $SYSCTL_CONF..."
    sudo sed -i '/vm.swappiness/d' $SYSCTL_CONF
    echo "vm.swappiness=$LOW_SWAPPINESS" | sudo tee -a $SYSCTL_CONF

    echo "→ Clearing swap (moving back to RAM)..."
    sudo swapoff -a && sudo swapon -a

    echo ""
    echo "Done. Current memory usage:"
    free -h
    ;;

  revert)
    echo "→ Restoring swappiness to $DEFAULT_SWAPPINESS..."
    sudo sysctl vm.swappiness=$DEFAULT_SWAPPINESS

    echo "→ Removing from $SYSCTL_CONF..."
    sudo sed -i '/vm.swappiness/d' $SYSCTL_CONF

    echo ""
    echo "Done. Current memory usage:"
    free -h
    ;;

  *)
    echo "Usage: $0 [apply|revert]"
    echo ""
    echo "  apply  — set swappiness=$LOW_SWAPPINESS, clear swap"
    echo "  revert — restore swappiness=$DEFAULT_SWAPPINESS"
    exit 1
    ;;
esac
