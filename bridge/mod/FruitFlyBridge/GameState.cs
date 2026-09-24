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
                return new JObject { ["ok"] = true, ["in_world"] = false };
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
                ["tiles"] = BuildTileGrid(player),
            };
            return result;
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
