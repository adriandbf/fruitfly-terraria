# Phase 1: tModLoader dedicated server on the Windows PC

Goal: the PC hosts a world; the Mac (and later the fly bot) joins it over the local network.
Done when the Mac and the PC can both be in the same world at once.

## Pinned versions
- tModLoader: v2026.07.3.0 (latest release when this was written, 2026-09-20).
  Release: https://github.com/tModLoader/tModLoader/releases/tag/v2026.07.3.0
- The Mac client and the PC server must run the same tModLoader version. Pin it in
  Steam on both machines (tModLoader > Properties > Betas) if you need to hold a version.
- Terraria itself: whatever Steam ships alongside that tModLoader release. Record it here once known: `TODO`.

## On the PC (Windows)
1. Install tModLoader from Steam (free tool, needs Terraria owned on the account). Launch it
   once so it downloads and sets up its files, then quit.
2. Find the install folder (Steam > tModLoader > Manage > Browse local files). It contains
   `start-tModLoaderServer.bat`.
3. Copy `serverconfig.txt` from this folder next to that script and change `password`.
4. Allow the port through Windows Firewall, private networks and local subnet only.
   In an admin PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "Terraria server 7777" -Direction Inbound -Protocol TCP `
     -LocalPort 7777 -Action Allow -Profile Private -RemoteAddress LocalSubnet
   ```
   The wifi network must be set to Private (Settings > Network > Properties), or the rule won't apply.
5. Start the server: `start-tModLoaderServer.bat -config serverconfig.txt`.
   It creates a small world on first run and prints `Server started`.
6. Find the PC's LAN address: `ipconfig` (the IPv4 address of the wifi/ethernet adapter).
   Give the PC a DHCP reservation on the router so it doesn't change.

## On the Mac
1. Install Terraria and tModLoader from Steam. Terraria is not currently in this Mac's
   Steam library, so it needs to be downloaded first.
2. Check the port is reachable before launching the game:
   ```bash
   bash server/check_connection.sh <PC-LAN-IP>
   ```
3. Launch tModLoader > Multiplayer > Join via IP > `<PC-LAN-IP>` port `7777`, enter the password.

## Report back
- The exact Terraria and tModLoader versions on both machines.
- Whether `check_connection.sh` passes, and whether both players show up in the same world.
- Round-trip feel over wifi (lag, rubber-banding). Phase 2 relies on this link.
