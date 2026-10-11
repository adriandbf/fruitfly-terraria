using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Newtonsoft.Json.Linq;
using Terraria;
using Terraria.ID;

namespace FruitFlyBridge
{
    /// <summary>
    /// A Nurse house designed by hand in the game (user, 2026-10-11), recorded
    /// and rebuilt exactly, relative to the arena mark:
    ///
    ///   scan_nurse_house {dx0, dy0, dx1, dy1}  -- records every tile, wall and
    ///       liquid in the box (tiles; dx from the arena mark's column, dy from
    ///       the arena floor row, down = +), the beds (their spawn tiles, by the
    ///       game's bed rule) and where the Nurse stands.
    ///   build_nurse_house {blueprint: ...}     -- clears the box and writes
    ///       the recorded tiles back verbatim (furniture frames included), sets
    ///       the spawn to the bed and puts the Nurse where she was.
    ///
    /// Cell = [dx, dy, tileType or -1, frameX, frameY, slope, halfBrick, wall,
    ///         liquidAmount, liquidType, actuated, tileColor, wallColor].
    /// Chests, signs and other tile entities are not recorded.
    /// </summary>
    public static class NurseBlueprint
    {
        private static bool ArenaOrigin(out int xa, out int s, out string error)
        {
            xa = s = 0;
            error = null;
            if (!Episode.ArenaPosition.HasValue)
            {
                error = "no arena mark (stand on the arena floor, --mark-arena)";
                return false;
            }
            var a = Episode.ArenaPosition.Value;
            var p = Main.LocalPlayer;
            xa = (int)((a.X + p.width / 2f) / 16f);
            s = (int)((a.Y + p.height) / 16f);
            return true;
        }

        private static Point? BedSpawn(int x, int y)
        {
            var t = Main.tile[x, y];
            if (!t.HasTile || t.TileType != TileID.Beds)
                return null;
            // The game's rule (Player.TileInteractionsUse, beds).
            int sx = x - t.TileFrameX / 18 + (t.TileFrameX >= 72 ? 5 : 2);
            int sy = y - (t.TileFrameY / 18) % 2 + 2;
            return new Point(sx, sy);
        }

        public static JObject Scan(JObject req)
        {
            if (!ArenaOrigin(out int xa, out int s, out string err))
                return new JObject { ["ok"] = false, ["error"] = err };
            int dx0 = (int?)req["dx0"] ?? -40, dx1 = (int?)req["dx1"] ?? 40;
            int dy0 = (int?)req["dy0"] ?? -2, dy1 = (int?)req["dy1"] ?? 60;
            if (!NurseBay.InWorld(xa + dx0, s + dy0) || !NurseBay.InWorld(xa + dx1, s + dy1))
                return new JObject { ["ok"] = false, ["error"] = "scan box is outside the world" };

            var cells = new JArray();
            var beds = new List<Point>();
            for (int x = xa + dx0; x <= xa + dx1; x++)
                for (int y = s + dy0; y <= s + dy1; y++)
                {
                    var t = Main.tile[x, y];
                    if (!t.HasTile && t.WallType == 0 && t.LiquidAmount == 0)
                        continue;
                    cells.Add(new JArray(x - xa, y - s, t.HasTile ? (int)t.TileType : -1, (int)t.TileFrameX, (int)t.TileFrameY,
                                         (int)t.Slope, t.IsHalfBlock ? 1 : 0, (int)t.WallType, (int)t.LiquidAmount, t.LiquidType,
                                         t.IsActuated ? 1 : 0, (int)t.TileColor, (int)t.WallColor));
                    var b = BedSpawn(x, y);
                    if (b.HasValue && !beds.Contains(b.Value))
                        beds.Add(b.Value);
                }

            var bedsJson = new JArray();
            foreach (var b in beds)
                bedsJson.Add(new JObject { ["dx"] = b.X - xa, ["dy"] = b.Y - s, ["valid_spawn"] = Player.CheckSpawn(b.X, b.Y) });
            var n = NurseBay.FindNurse();
            JObject nurse = null;
            if (n != null)
            {
                int nx = (int)(n.Center.X / 16f), ny = (int)((n.position.Y + n.height) / 16f);
                if (nx >= xa + dx0 && nx <= xa + dx1 && ny >= s + dy0 && ny <= s + dy1)
                    nurse = new JObject { ["dx"] = nx - xa, ["dy"] = ny - s };
            }
            int chests = 0;
            foreach (var c in Main.chest)
                if (c != null && c.x >= xa + dx0 && c.x <= xa + dx1 && c.y >= s + dy0 && c.y <= s + dy1)
                    chests++;
            return new JObject
            {
                ["ok"] = beds.Count > 0,
                ["error"] = beds.Count > 0 ? null : "no bed in the scan box",
                ["box"] = new JObject { ["dx0"] = dx0, ["dy0"] = dy0, ["dx1"] = dx1, ["dy1"] = dy1 },
                ["origin"] = new JObject { ["x"] = xa, ["y"] = s },
                ["cells"] = cells,
                ["beds"] = bedsJson,
                ["nurse"] = nurse,
                ["chests_not_recorded"] = chests,
            };
        }

        public static JObject Build(JObject req, JObject bp, Vector2 origin, int demolished)
        {
            var p = Main.LocalPlayer;
            int xa = (int)((origin.X + p.width / 2f) / 16f), s = (int)((origin.Y + p.height) / 16f);
            var box = (JObject)bp["box"];
            int x0 = xa + (int)box["dx0"], x1 = xa + (int)box["dx1"], y0 = s + (int)box["dy0"], y1 = s + (int)box["dy1"];
            if (!NurseBay.InWorld(x0, y0) || !NurseBay.InWorld(x1, y1))
                return new JObject { ["ok"] = false, ["error"] = "blueprint box would be outside the world" };
            var beds = bp["beds"] as JArray;
            if (beds == null || beds.Count == 0)
                return new JObject { ["ok"] = false, ["error"] = "blueprint has no bed" };

            // Clear the box (chests, signs, liquids too), then write the cells back.
            Setup.ClearArea(x0, y0, x1, y1);
            int written = 0;
            foreach (var tok in (JArray)bp["cells"])
            {
                var c = (JArray)tok;
                int x = xa + (int)c[0], y = s + (int)c[1];
                if (x < x0 || x > x1 || y < y0 || y > y1)
                    continue;
                var t = Main.tile[x, y];
                int type = (int)c[2];
                if (type >= 0)
                {
                    t.HasTile = true;
                    t.TileType = (ushort)type;
                    t.TileFrameX = (short)(int)c[3];
                    t.TileFrameY = (short)(int)c[4];
                    t.Slope = (SlopeType)(int)c[5];
                    t.IsHalfBlock = (int)c[6] != 0;
                    t.IsActuated = (int)c[10] != 0;
                    t.TileColor = (byte)(int)c[11];
                }
                t.WallType = (ushort)(int)c[7];
                t.LiquidAmount = (byte)(int)c[8];
                t.LiquidType = (int)c[9];
                t.WallColor = (byte)(int)c[12];
                written++;
            }
            // Let the edges join up with the world around the box.
            WorldGen.RangeFrame(x0 - 1, y0 - 1, x1 + 1, y0);
            WorldGen.RangeFrame(x0 - 1, y1, x1 + 1, y1 + 1);
            WorldGen.RangeFrame(x0 - 1, y0, x0, y1);
            WorldGen.RangeFrame(x1, y0, x1 + 1, y1);

            var bed = (JObject)beds[0];
            var sp = new Point(xa + (int)bed["dx"], s + (int)bed["dy"]);
            NurseBay.SpawnPoint = sp;
            NurseBay.House = sp;
            NurseBay.HouseSurface = s;
            NurseBay.Box = new Rectangle(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
            var ns = bp["nurse"] as JObject;
            NurseBay.NurseSpot = ns != null ? new Point(xa + (int)ns["dx"], s + (int)ns["dy"]) : new Point(sp.X + 3, sp.Y);
            NurseBay.Room = FloodRoom(sp);

            var spawn = NurseBay.SetSpawnToBed();
            var nurse = NurseBay.EnsureNurse();
            return new JObject
            {
                ["ok"] = spawn.HasValue,
                ["error"] = spawn.HasValue ? null : NurseBay.SpawnError,
                ["blueprint"] = true,
                ["cells_written"] = written,
                ["room_valid"] = WorldGen.StartRoomCheck(sp.X, sp.Y - 1) && WorldGen.RoomNeeds(NPCID.Nurse),
                ["house"] = new JObject { ["x"] = sp.X, ["y"] = sp.Y },
                ["shaft_top_y"] = s,
                ["old_house_tiles_removed"] = demolished,
                ["spawn"] = spawn.HasValue ? new JObject { ["x"] = spawn.Value.X, ["y"] = spawn.Value.Y } : null,
                ["nurse"] = nurse,
                ["room"] = NurseBay.RectJson(NurseBay.Room.Value),
                ["restore"] = NurseBay.RestoreJson(),
            };
        }

        /// <summary>The bed's room: open cells reachable from just above the bed's
        /// spawn tile without crossing a solid tile or platform. Falls back to a
        /// small box around the bed if it leaks (not enclosed).</summary>
        private static Rectangle FloodRoom(Point sp)
        {
            var seen = new HashSet<Point>();
            var todo = new Stack<Point>();
            todo.Push(new Point(sp.X, sp.Y - 1));
            int minX = sp.X, maxX = sp.X, minY = sp.Y - 1, maxY = sp.Y - 1;
            while (todo.Count > 0)
            {
                var q = todo.Pop();
                if (seen.Contains(q) || !NurseBay.InWorld(q.X, q.Y))
                    continue;
                var t = Main.tile[q.X, q.Y];
                if (t.HasTile && Main.tileSolid[t.TileType] && !t.IsActuated)
                    continue;
                seen.Add(q);
                if (seen.Count > 3000)
                    return new Rectangle(sp.X - 6, sp.Y - 6, 13, 7);
                minX = Math.Min(minX, q.X); maxX = Math.Max(maxX, q.X);
                minY = Math.Min(minY, q.Y); maxY = Math.Max(maxY, q.Y);
                todo.Push(new Point(q.X + 1, q.Y));
                todo.Push(new Point(q.X - 1, q.Y));
                todo.Push(new Point(q.X, q.Y + 1));
                todo.Push(new Point(q.X, q.Y - 1));
            }
            return new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
        }
    }
}
