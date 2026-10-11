using System;
using Microsoft.Xna.Framework;
using Newtonsoft.Json.Linq;
using Terraria;
using Terraria.Audio;
using Terraria.DataStructures;
using Terraria.ID;

namespace FruitFlyBridge
{
    /// <summary>
    /// The Nurse healing protocol (user design, 2026-10-11): an underground
    /// house below the arena with the fly's bed (its spawn point, so the Magic
    /// Mirror lands there) and the Nurse boxed in next to it under a SOLID
    /// roof; the fly's side has a PLATFORM roof and a shaft up to the arena,
    /// so it can heal and fly straight back out. The Moon Lord teleports onto
    /// a player ~2,300 px away (live logs), so it will follow the fly down:
    /// the Nurse's box keeps her out of reach.
    ///
    /// Layout (tiles; F = house floor row, X = house centre column):
    ///
    ///   shaft (cleared up to the arena floor, platform cap at the top)
    ///   |..........|
    ///   ============ ######        = platforms   # = gray brick
    ///   #  fly room  # nurse #
    ///   #   [bed]    #  (N)  #
    ///   ######################  row F
    ///    X-7 .. X-1  X  X+1..X+5  X+6
    ///
    /// The heal itself (nurse_heal) does what the Nurse's "Heal" chat button
    /// does -- same price rule, charged with BuyItem, same debuff removal --
    /// because the fly can't click through a chat window. It requires the
    /// Nurse alive and within NurseRangeTiles.
    /// </summary>
    public static class NurseBay
    {
        public static Point? House;          // (X, F) in tiles
        public const int NurseRangeTiles = 15;

        private static bool InWorld(int x, int y) => x > 10 && y > 10 && x < Main.maxTilesX - 10 && y < Main.maxTilesY - 10;

        private static void Solid(int x, int y)
        {
            var t = Main.tile[x, y];
            if (t.HasTile && t.TileType == TileID.GrayBrick)
                return;
            if (t.HasTile)
                WorldGen.KillTile(x, y, false, false, true);
            WorldGen.PlaceTile(x, y, TileID.GrayBrick, true, true);
        }

        private static void Platform(int x, int y)
        {
            var t = Main.tile[x, y];
            if (t.HasTile && t.TileType == TileID.Platforms)
                return;
            if (t.HasTile)
                WorldGen.KillTile(x, y, false, false, true);
            WorldGen.PlaceTile(x, y, TileID.Platforms, true, true, -1, 0);
        }

        /// <summary>build_nurse_house {depth_tiles=40, offset_x_tiles=0, x?, y?}:
        /// x/y = the arena mark (player top-left, px); default the current arena
        /// mark, else where the player stands. Builds the house, sets the fly's
        /// spawn to the bed, puts the Nurse in her box.</summary>
        public static JObject Build(JObject req)
        {
            var p = Main.LocalPlayer;
            var origin = Episode.ArenaPosition ?? p.position;
            if (req["x"] != null && req["y"] != null)
                origin = new Vector2((float)req["x"], (float)req["y"]);
            int depth = (int?)req["depth_tiles"] ?? 40;
            int offset = (int?)req["offset_x_tiles"] ?? 0;
            int X = (int)((origin.X + p.width / 2f) / 16f) + offset;
            int surface = (int)((origin.Y + p.height) / 16f);   // arena floor row (ground under the mark)
            int F = surface + depth;
            if (!InWorld(X - 9, F - 8) || !InWorld(X + 8, F + 2))
                return new JObject { ["ok"] = false, ["error"] = "house would be outside the world" };

            // Clear the house box and the shaft (fly side) up to the arena floor.
            Setup.ClearArea(X - 8, F - 7, X + 7, F);
            Setup.ClearArea(X - 6, surface, X - 1, F - 7);

            // Shell.
            for (int x = X - 7; x <= X + 6; x++)
                Solid(x, F);
            for (int y = F - 6; y <= F - 1; y++)
            {
                Solid(X - 7, y);
                Solid(X, y);        // divider: the Nurse stays on her side
                Solid(X + 6, y);
            }
            for (int x = X - 6; x <= X - 1; x++)
                Platform(x, F - 6); // fly room roof: fly straight up through it
            for (int x = X + 1; x <= X + 5; x++)
                Solid(x, F - 6);    // Nurse roof: solid
            // Cap the shaft where it meets the arena floor with platforms (passable
            // both ways), so the arena floor stays walkable.
            for (int x = X - 6; x <= X - 1; x++)
                Platform(x, surface);
            // Shaft sides: solid, so nothing spills sideways into the dirt.
            for (int y = surface + 1; y <= F - 7; y++)
            {
                Solid(X - 7, y);
                Solid(X, y);
            }

            // Background walls, light, the bed.
            for (int x = X - 6; x <= X + 5; x++)
                for (int y = F - 5; y <= F - 1; y++)
                    if (x != X)
                        WorldGen.PlaceWall(x, y, WallID.GrayBrick, true);
            WorldGen.PlaceTile(X - 1, F - 4, TileID.Torches, true);
            WorldGen.PlaceTile(X + 5, F - 4, TileID.Torches, true);
            bool bed = WorldGen.PlaceObject(X - 4, F - 1, TileID.Beds, true, 0);
            WorldGen.RangeFrame(X - 9, surface - 1, X + 8, F + 1);

            House = new Point(X, F);
            var spawn = SetSpawnToBed();
            var nurse = EnsureNurse();
            return new JObject
            {
                ["ok"] = bed && spawn.HasValue,
                ["error"] = !bed ? "bed could not be placed" : (!spawn.HasValue ? "bed not found for spawn" : null),
                ["house"] = new JObject { ["x"] = X, ["y"] = F },
                ["shaft_top_y"] = surface,
                ["spawn"] = spawn.HasValue ? new JObject { ["x"] = spawn.Value.X, ["y"] = spawn.Value.Y } : null,
                ["nurse"] = nurse,
            };
        }

        /// <summary>Spawn point = the bed, using the game's own bed rule
        /// (left column + 2, row below the bed), so the Magic Mirror lands there.</summary>
        private static Point? SetSpawnToBed()
        {
            if (!House.HasValue)
                return null;
            var h = House.Value;
            for (int x = h.X - 6; x <= h.X - 1; x++)
                for (int y = h.Y - 3; y <= h.Y - 1; y++)
                {
                    var t = Main.tile[x, y];
                    if (t.HasTile && t.TileType == TileID.Beds)
                    {
                        // Top-left tile of the bed: scan from its frame.
                        int left = x - (t.TileFrameX % 72) / 18;
                        int top = y - t.TileFrameY / 18;
                        int sx = left + 2, sy = top + 2;
                        if (Player.CheckSpawn(sx, sy))
                        {
                            Main.LocalPlayer.ChangeSpawn(sx, sy);
                            return new Point(sx, sy);
                        }
                    }
                }
            return null;
        }

        public static NPC FindNurse()
        {
            foreach (var n in Main.npc)
                if (n != null && n.active && n.type == NPCID.Nurse)
                    return n;
            return null;
        }

        /// <summary>The Nurse exists, lives in her box, at full health.</summary>
        public static JObject EnsureNurse()
        {
            if (!House.HasValue)
                return new JObject { ["ok"] = false, ["error"] = "no nurse house (build_nurse_house or pass x/y)" };
            var h = House.Value;
            var n = FindNurse();
            bool spawned = false;
            if (n == null)
            {
                int idx = NPC.NewNPC(new EntitySource_Misc("FruitFlyBridge.NurseBay"), (h.X + 3) * 16 + 8, (h.Y - 1) * 16, NPCID.Nurse);
                if (idx < 0 || idx >= Main.maxNPCs)
                    return new JObject { ["ok"] = false, ["error"] = "could not spawn the Nurse" };
                n = Main.npc[idx];
                spawned = true;
            }
            n.position = new Vector2((h.X + 3) * 16 + 8 - n.width / 2f, h.Y * 16 - n.height);
            n.velocity = Vector2.Zero;
            n.life = n.lifeMax;
            n.homeTileX = h.X + 3;
            n.homeTileY = h.Y - 1;
            n.homeless = false;
            return new JObject { ["ok"] = true, ["spawned"] = spawned, ["hp"] = n.life, ["max_hp"] = n.lifeMax };
        }

        /// <summary>nurse_ready {x?, y?}: restore the house position (tiles; e.g.
        /// after a mod rebuild), re-set the spawn to the bed, ensure the Nurse.</summary>
        public static JObject Ready(JObject req)
        {
            if (req["x"] != null && req["y"] != null)
                House = new Point((int)req["x"], (int)req["y"]);
            var spawn = SetSpawnToBed();
            var nurse = EnsureNurse();
            nurse["spawn_set"] = spawn.HasValue;
            return nurse;
        }

        /// <summary>nurse_heal: the Nurse's Heal button, for the fly. Same price
        /// rule as the game's first steps (missing HP + 100 per removable debuff,
        /// times the progression multiplier; the game then applies a few more
        /// adjustments we don't copy -- the character carries 999 platinum), paid with BuyItem; restores full HP and removes
        /// the debuffs the Nurse can remove (not Potion Sickness etc.).</summary>
        public static JObject Heal(JObject req)
        {
            var p = Main.LocalPlayer;
            var n = FindNurse();
            if (n == null)
                return new JObject { ["ok"] = false, ["error"] = "no Nurse" };
            float dist = Vector2.Distance(p.Center, n.Center) / 16f;
            if (dist > NurseRangeTiles)
                return new JObject { ["ok"] = false, ["error"] = "too far from the Nurse", ["distance_tiles"] = dist };
            int missing = p.statLifeMax2 - p.statLife;
            int price = missing;
            for (int j = 0; j < p.buffType.Length; j++)
            {
                int b = p.buffType[j];
                if (b > 0 && Main.debuff[b] && p.buffTime[j] > 60 && !BuffID.Sets.NurseCannotRemoveDebuff[b])
                    price += 100;
            }
            if (NPC.downedGolemBoss) price *= 200;
            else if (NPC.downedPlantBoss) price *= 150;
            else if (NPC.downedMechBossAny) price *= 100;
            else if (Main.hardMode) price *= 60;
            else if (NPC.downedBoss3 || NPC.downedQueenBee) price *= 25;
            else if (NPC.downedBoss2) price *= 10;
            else if (NPC.downedBoss1) price *= 3;
            if (price <= 0)
                return new JObject { ["ok"] = true, ["healed"] = 0, ["price"] = 0 };
            if (!p.BuyItem(price))
                return new JObject { ["ok"] = false, ["error"] = "can't afford", ["price"] = price };
            SoundEngine.PlaySound(SoundID.Item4, p.Center);
            p.HealEffect(missing, true);
            p.statLife = p.statLifeMax2;
            for (int l = 0; l < p.buffType.Length; l++)
            {
                int b = p.buffType[l];
                if (b > 0 && Main.debuff[b] && p.buffTime[l] > 0 && !BuffID.Sets.NurseCannotRemoveDebuff[b])
                {
                    p.DelBuff(l);
                    l = -1;
                }
            }
            return new JObject { ["ok"] = true, ["healed"] = missing, ["price"] = price, ["distance_tiles"] = dist };
        }

        public static JObject ToJson()
        {
            if (!House.HasValue)
                return null;
            var n = FindNurse();
            return new JObject
            {
                ["x"] = House.Value.X,
                ["y"] = House.Value.Y,
                ["nurse_alive"] = n != null,
                ["nurse_hp"] = n?.life ?? 0,
                ["spawn_x"] = Main.LocalPlayer.SpawnX,
                ["spawn_y"] = Main.LocalPlayer.SpawnY,
            };
        }
    }
}
