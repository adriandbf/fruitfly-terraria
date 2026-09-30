using Newtonsoft.Json.Linq;
using Terraria;

namespace FruitFlyBridge
{
    /// <summary>
    /// Builds the JSON state snapshot returned by "get_state". Kept small and
    /// flat for v0 (Phase 2's dumb bot); richer/derived features belong in the
    /// Python-side encoder, not here.
    ///
    /// UNTESTED against a real tModLoader build — see BridgeSystem.cs. Field
    /// names below (Main.npc, Main.tile, Main.tileSolid) are from memory of
    /// the modding API and may need small fixes once this actually compiles.
    /// </summary>
    public static class GameState
    {
        private const int NpcRadius = 800; // pixels (~50 tiles)
        private const int TileHalfWidth = 15;
        private const int TileHalfHeight = 8;

        public static JObject BuildSnapshot()
        {
            if (Main.gameMenu || Main.LocalPlayer == null)
            {
                // Temporary diagnostics: in_world came back false when the
                // player insisted they were standing in-world, so surface the
                // exact flags instead of guessing at another fix blind.
                return new JObject
                {
                    ["ok"] = true,
                    ["in_world"] = false,
                    ["debug"] = new JObject
                    {
                        ["game_menu"] = Main.gameMenu,
                        ["local_player_null"] = Main.LocalPlayer == null,
                        ["my_player_index"] = Main.myPlayer,
                        ["net_mode"] = Main.netMode,
                    },
                };
            }

            var player = Main.LocalPlayer;

            var result = new JObject
            {
                ["ok"] = true,
                ["in_world"] = true,
                ["player"] = new JObject
                {
                    ["x"] = player.position.X,
                    ["y"] = player.position.Y,
                    ["direction"] = player.direction,
                    ["hp"] = player.statLife,
                    ["max_hp"] = player.statLifeMax,
                    ["mana"] = player.statMana,
                    ["max_mana"] = player.statManaMax,
                    ["on_ground"] = player.velocity.Y == 0f,
                },
                ["npcs"] = BuildNpcArray(player),
                ["players"] = BuildOtherPlayersArray(player),
                ["tiles"] = BuildTileGrid(player),
                // Temporary diagnostics: a held "right" produced zero
                // movement in a live multiplayer test, with manual
                // keyboard control confirmed working and no UI open (so
                // not a Main.hasFocus / chat / menu gate). Surfacing the
                // actual applied control values and known movement-
                // blocking flags instead of guessing at another fix blind.
                // isOperatingAnotherEntity is a real Player field (mounts/
                // vehicles zero all movement controls AFTER ProcessTriggers
                // runs, which would exactly explain this). frozen/stoned/
                // CCed are from memory of the Player class, not verified
                // against this exact tModLoader version -- if any of these
                // don't compile, that's the build error to fix, drop the
                // offending line(s) rather than guess at a different name.
                ["debug"] = new JObject
                {
                    ["has_focus"] = Main.hasFocus,
                    ["control_left"] = player.controlLeft,
                    ["control_right"] = player.controlRight,
                    ["velocity_x"] = player.velocity.X,
                    ["is_operating_another_entity"] = player.isOperatingAnotherEntity,
                    ["mount_active"] = player.mount != null && player.mount.Active,
                    ["frozen"] = player.frozen,
                    ["stoned"] = player.stoned,
                    ["cc_immune"] = player.CCed,
                },
            };
            return result;
        }

        // Other human players in the world (e.g. the PC's player) -- NOT
        // included in "npcs". Added for Phase 4: the encoder needs the
        // human's position for a "walk toward the human" drive, and
        // Main.npc never contains other players.
        private static JArray BuildOtherPlayersArray(Player player)
        {
            var players = new JArray();
            foreach (var other in Main.player)
            {
                if (other == null || !other.active || other.whoAmI == player.whoAmI)
                {
                    continue;
                }

                players.Add(new JObject
                {
                    ["name"] = other.name,
                    ["x"] = other.position.X,
                    ["y"] = other.position.Y,
                    ["hp"] = other.statLife,
                    ["max_hp"] = other.statLifeMax,
                });
            }
            return players;
        }

        private static JArray BuildNpcArray(Player player)
        {
            var npcs = new JArray();
            foreach (var npc in Main.npc)
            {
                if (npc == null || !npc.active)
                {
                    continue;
                }

                var dx = npc.position.X - player.position.X;
                var dy = npc.position.Y - player.position.Y;
                if (dx * dx + dy * dy > NpcRadius * NpcRadius)
                {
                    continue;
                }

                npcs.Add(new JObject
                {
                    ["type"] = npc.type,
                    ["name"] = npc.FullName,
                    ["x"] = npc.position.X,
                    ["y"] = npc.position.Y,
                    ["hp"] = npc.life,
                    ["hostile"] = npc.friendly ? false : true,
                });
            }
            return npcs;
        }

        // Flat row-major array of tile type ids, -1 outside the world, centred
        // on the player's tile. (2*TileHalfWidth+1) columns by
        // (2*TileHalfHeight+1) rows. "solid" is looked up from vanilla's own
        // Main.tileSolid table so this stays correct across tile-type additions.
        private static JObject BuildTileGrid(Player player)
        {
            int cx = (int)(player.Center.X / 16f);
            int cy = (int)(player.Center.Y / 16f);

            var types = new JArray();
            var solid = new JArray();

            for (int ty = cy - TileHalfHeight; ty <= cy + TileHalfHeight; ty++)
            {
                for (int tx = cx - TileHalfWidth; tx <= cx + TileHalfWidth; tx++)
                {
                    if (tx < 0 || ty < 0 || tx >= Main.maxTilesX || ty >= Main.maxTilesY)
                    {
                        types.Add(-1);
                        solid.Add(false);
                        continue;
                    }

                    var tile = Main.tile[tx, ty];
                    if (tile == null || !tile.HasTile)
                    {
                        types.Add(-1);
                        solid.Add(false);
                        continue;
                    }

                    types.Add(tile.TileType);
                    solid.Add(Main.tileSolid[tile.TileType]);
                }
            }

            return new JObject
            {
                ["origin_tx"] = cx - TileHalfWidth,
                ["origin_ty"] = cy - TileHalfHeight,
                ["width"] = TileHalfWidth * 2 + 1,
                ["height"] = TileHalfHeight * 2 + 1,
                ["types"] = types,
                ["solid"] = solid,
            };
        }
    }
}
