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
    /// Layout v2 (2026-10-11; v1's fly room was ~50 tiles with its frame,
    /// under the game's 60-tile house minimum, so the bed could not be a
    /// spawn point and the Magic Mirror went to the world spawn). Tiles;
    /// F = house floor row, X = divider column:
    ///
    ///   shaft X-10..X-1 (cleared up to the arena floor, platform cap there)
    ///   #==========#######        = platforms   # = gray brick
    ///   #  fly room  # nurse #     interiors 10x6 and 6x6, both valid houses
    ///   # bed ch tbl #  (N)  #     (walls, torch, chair, table, bed)
    ///   ######################  row F
    ///   X-11        X       X+7
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
            Setup.ClearArea(X - 12, F - 8, X + 8, F);
            Setup.ClearArea(X - 10, surface, X - 1, F - 8);

            // Shell.
            for (int x = X - 11; x <= X + 7; x++)
                Solid(x, F);
            for (int y = F - 7; y <= F - 1; y++)
            {
                Solid(X - 11, y);
                Solid(X, y);        // divider: the Nurse stays on her side
                Solid(X + 7, y);
            }
            for (int x = X - 10; x <= X - 1; x++)
                Platform(x, F - 7); // fly room roof: fly straight up through it
            for (int x = X + 1; x <= X + 6; x++)
                Solid(x, F - 7);    // Nurse roof: solid
            // Cap the shaft where it meets the arena floor with platforms (passable
            // both ways), so the arena floor stays walkable.
            for (int x = X - 10; x <= X - 1; x++)
                Platform(x, surface);
            // Shaft sides: solid, so nothing spills sideways into the dirt.
            for (int y = surface + 1; y <= F - 8; y++)
            {
                Solid(X - 11, y);
                Solid(X, y);
            }

            // Background walls, light, furniture (a valid house in both rooms).
            for (int x = X - 10; x <= X + 6; x++)
                for (int y = F - 6; y <= F - 1; y++)
                    if (x != X)
                        WorldGen.PlaceWall(x, y, WallID.GrayBrick, true);
            WorldGen.PlaceTile(X - 1, F - 5, TileID.Torches, true);
            WorldGen.PlaceTile(X + 6, F - 5, TileID.Torches, true);
            bool bed = WorldGen.PlaceObject(X - 8, F - 1, TileID.Beds, true, 0);       // X-9..X-6
            WorldGen.PlaceObject(X - 5, F - 1, TileID.Chairs, true, 0);
            WorldGen.PlaceObject(X - 3, F - 1, TileID.Tables, true, 0);                // X-4..X-2
            WorldGen.PlaceObject(X + 2, F - 1, TileID.Chairs, true, 0);
            WorldGen.PlaceObject(X + 5, F - 1, TileID.Tables, true, 0);                // X+4..X+6
            WorldGen.RangeFrame(X - 13, surface - 1, X + 9, F + 1);

            House = new Point(X, F);
            var spawn = SetSpawnToBed();
            var nurse = EnsureNurse();
            return new JObject
            {
                ["ok"] = bed && spawn.HasValue,
                ["error"] = !bed ? "bed could not be placed" : (!spawn.HasValue ? SpawnError : null),
                ["fly_room_valid"] = WorldGen.StartRoomCheck(X - 5, F - 3),
                ["house"] = new JObject { ["x"] = X, ["y"] = F },
                ["shaft_top_y"] = surface,
                ["spawn"] = spawn.HasValue ? new JObject { ["x"] = spawn.Value.X, ["y"] = spawn.Value.Y } : null,
                ["nurse"] = nurse,
            };
        }

        /// <summary>Spawn point = the bed, using the game's own bed rule
        /// (left column + 2, row below the bed), so the Magic Mirror lands there.</summary>
        public static string SpawnError;

        private static Point? SetSpawnToBed()
        {
            SpawnError = null;
            if (!House.HasValue)
            {
                SpawnError = "no house";
                return null;
            }
            var h = House.Value;
            for (int x = h.X - 10; x <= h.X - 1; x++)
                for (int y = h.Y - 3; y <= h.Y - 1; y++)
                {
                    var t = Main.tile[x, y];
                    if (t.HasTile && t.TileType == TileID.Beds)
                    {
                        // The game's own rule (Player.TileInteractionsUse, beds):
                        // x - frameX/18, +2 facing left / +5 facing right; top + 2.
                        int sx = x - t.TileFrameX / 18 + (t.TileFrameX >= 72 ? 5 : 2);
                        int sy = y - (t.TileFrameY / 18) % 2 + 2;
                        if (Player.CheckSpawn(sx, sy))
                        {
                            Main.LocalPlayer.ChangeSpawn(sx, sy);
                            return new Point(sx, sy);
                        }
                        SpawnError = WorldGen.StartRoomCheck(sx, sy - 1)
                            ? "bed spawn blocked (something solid above the bed)"
                            : "bed is not in a valid house (game's room check failed)";
                        return null;
                    }
                }
            SpawnError = "bed not found";
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
                int idx = NPC.NewNPC(new EntitySource_Misc("FruitFlyBridge.NurseBay"), (h.X + 4) * 16 + 8, (h.Y - 1) * 16, NPCID.Nurse);
                if (idx < 0 || idx >= Main.maxNPCs)
                    return new JObject { ["ok"] = false, ["error"] = "could not spawn the Nurse" };
                n = Main.npc[idx];
                spawned = true;
            }
            n.position = new Vector2((h.X + 4) * 16 + 8 - n.width / 2f, h.Y * 16 - n.height);
            n.velocity = Vector2.Zero;
            n.life = n.lifeMax;
            n.homeTileX = h.X + 4;
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
            if (!spawn.HasValue)
            {
                nurse["ok"] = false;
                nurse["error"] = "spawn not set to the bed: " + SpawnError;
            }
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
                ["spawn_at_bed"] = Main.LocalPlayer.SpawnX >= House.Value.X - 11 && Main.LocalPlayer.SpawnX <= House.Value.X
                                   && Math.Abs(Main.LocalPlayer.SpawnY - House.Value.Y) <= 2,
            };
        }
    }
}
