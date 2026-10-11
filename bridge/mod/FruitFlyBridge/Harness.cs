using System;
using System.Collections.Concurrent;
using System.Threading;
using Microsoft.Xna.Framework;
using Newtonsoft.Json.Linq;
using Terraria;
using Terraria.DataStructures;
using Terraria.ID;
using Terraria.ModLoader;

namespace FruitFlyBridge
{
    /// <summary>
    /// Goal 2 (Moon Lord) episode harness: per-episode counters plus the
    /// commands that set up and tear down a fight (reset_episode,
    /// spawn_npc, spawn_projectile, select_slot, set_arena, set_keep_focus).
    ///
    /// Everything that mutates the world runs on the GAME thread: the socket
    /// handler enqueues a closure and blocks (up to TimeoutMs) until
    /// HarnessSystem.PostUpdateEverything has run it. Spawning NPCs or
    /// projectiles from the socket thread would race the game loop.
    /// </summary>
    public static class Episode
    {
        public static int Ticks;          // game ticks (60/s) since last reset
        public static int Hits;           // times the player was hurt
        public static int DamageTaken;    // sum of HurtInfo.Damage
        public static int Deaths;
        public static bool BossKilled;    // Moon Lord core killed since reset
        public static int BossKillTick = -1;

        public static Vector2? ArenaPosition; // where reset_episode puts the player
        public static volatile bool KeepFocus; // see HarnessSystem.PostUpdateInput
        // Human demo recording: movement keys come from the real keyboard
        // (the bridge only adds `use` + aim when it holds use, e.g. the whip
        // assist). get_state.debug reports the human's actual controls.
        public static volatile bool PassthroughMovement;

        public static void Reset()
        {
            Ticks = 0;
            Hits = 0;
            DamageTaken = 0;
            Deaths = 0;
            BossKilled = false;
            BossKillTick = -1;
        }

        public static JObject ToJson()
        {
            return new JObject
            {
                ["ticks"] = Ticks,
                ["seconds"] = Ticks / 60.0,
                ["hits"] = Hits,
                ["damage_taken"] = DamageTaken,
                ["deaths"] = Deaths,
                ["boss_killed"] = BossKilled,
                ["boss_kill_tick"] = BossKillTick,
                ["keep_focus"] = KeepFocus,
                ["passthrough"] = PassthroughMovement,
                ["arena"] = ArenaPosition.HasValue
                    ? new JObject { ["x"] = ArenaPosition.Value.X, ["y"] = ArenaPosition.Value.Y }
                    : null,
                ["full_gravity_y"] = FullGravityY(),
                ["nurse_house"] = NurseBay.ToJson(),
                ["world_surface_y"] = Main.worldSurface * 16.0,
                ["world_tiles"] = new JObject { ["x"] = Main.maxTilesX, ["y"] = Main.maxTilesY },
                ["world_name"] = Main.worldName,
                ["world_id"] = Main.worldID,
            };
        }

        /// <summary>
        /// World y (px) above which gravity starts to drop toward space. Vanilla
        /// Player.Update scales gravity by (y/16 - (60 + 10*(maxTilesX/4200)^2)) /
        /// (worldSurface/6), clamped to [0.25, 1]; this is where that factor
        /// reaches 1. Larger worlds put it further above the surface. (Formula
        /// from the 1.4.4 game code as remembered, not read from source here:
        /// the logged player.gravity is the ground truth.)
        /// </summary>
        public static double FullGravityY()
        {
            double n = Main.maxTilesX / 4200.0;
            n *= n;
            return (60.0 + 10.0 * n + Main.worldSurface / 6.0) * 16.0;
        }

        public static bool IsMoonLordPart(int type)
        {
            return type == NPCID.MoonLordCore || type == NPCID.MoonLordHead
                || type == NPCID.MoonLordHand || type == NPCID.MoonLordFreeEye
                || type == NPCID.MoonLordLeechBlob;
        }
    }

    /// <summary>
    /// Single-player Terraria pauses when its window loses focus (seen live
    /// 2026-10-08: the --keep-focus test sat paused). Forcing Main.hasFocus
    /// inside the update loop can't help if the loop itself is paused, so
    /// this sets FNA's Game.IsActive back to true from the socket thread on
    /// every bridge request (10 Hz), via reflection on its non-public
    /// setter (or the auto-property backing field). FNA only changes
    /// IsActive on a window focus event, so it then stays true. UNTESTED.
    /// </summary>
    public static class FocusHack
    {
        private static System.Reflection.MethodInfo _setter;
        private static System.Reflection.FieldInfo _field;
        private static bool _looked;

        public static bool Available
        {
            get
            {
                Lookup();
                return _setter != null || _field != null;
            }
        }

        private static void Lookup()
        {
            if (_looked)
            {
                return;
            }
            _looked = true;
            var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public
                        | System.Reflection.BindingFlags.NonPublic;
            var t = typeof(Microsoft.Xna.Framework.Game);
            _setter = t.GetProperty("IsActive", flags)?.GetSetMethod(true);
            if (_setter == null)
            {
                _field = t.GetField("<IsActive>k__BackingField", flags);
            }
        }

        public static void ForceActive()
        {
            var game = Main.instance;
            if (game == null || game.IsActive)
            {
                return;
            }
            Lookup();
            try
            {
                if (_setter != null)
                {
                    _setter.Invoke(game, new object[] { true });
                }
                else
                {
                    _field?.SetValue(game, true);
                }
            }
            catch
            {
                // best effort; get_state.debug.has_focus shows whether it worked
            }
        }
    }

    public class HarnessSystem : ModSystem
    {
        private const int TimeoutMs = 3000;

        private class Job
        {
            public Func<JObject> Work;
            public JObject Result;
            public readonly ManualResetEventSlim Done = new ManualResetEventSlim(false);
        }

        private static readonly ConcurrentQueue<Job> Jobs = new ConcurrentQueue<Job>();

        /// <summary>Called from the socket thread. Runs `work` on the game
        /// thread and returns its result, or an error if the game loop
        /// didn't get to it in time (in a menu, paused, or not updating).</summary>
        public static JObject RunOnGameThread(Func<JObject> work)
        {
            var job = new Job { Work = work };
            Jobs.Enqueue(job);
            if (!job.Done.Wait(TimeoutMs))
            {
                return new JObject
                {
                    ["ok"] = false,
                    ["error"] = $"game thread did not run the command within {TimeoutMs}ms " +
                                "(in a menu, game paused, or not updating while unfocused?)",
                };
            }
            return job.Result;
        }

        /// <summary>The arena mark and Nurse house are per world: forget them when
        /// the world unloads (live 2026-10-11: switching to a new world without
        /// restarting the game kept the old world's mark, so resets teleported the
        /// player to those coordinates, underground in the new world).</summary>
        public override void OnWorldUnload()
        {
            Episode.ArenaPosition = null;
            NurseBay.House = null;
        }

        public override void PostUpdateInput()
        {
            // Vanilla only feeds controls to the player while Main.hasFocus
            // is true (the live finding in CLAUDE.md: the character froze
            // whenever another app was frontmost). hasFocus is recomputed
            // from the window every frame BEFORE input handling; this hook
            // runs right after PlayerInput.UpdateInput() and before players
            // update, so forcing it here should let our held keys through.
            // Opt-in (set_keep_focus) and only while the bridge is
            // connected. UNTESTED live.
            if (Episode.KeepFocus && BridgeSystem.IsClientConnected && !Main.gameMenu)
            {
                Main.hasFocus = true;
            }

        }

        /// <summary>
        /// Aim (set_action aim_x/aim_y, world pixels): point the cursor at it
        /// so item use -- the whip assist -- swings that way. Done here, right
        /// before players update, rather than after input handling: input
        /// handling is followed by the zoom conversions that rewrite
        /// Main.mouseX/Y from the real mouse, which would undo it. During the
        /// world update Main.MouseWorld = screenPosition + mouse, so this is
        /// zoom-independent. Also published as debug.aim_world for checking.
        /// </summary>
        public override void PreUpdatePlayers()
        {
            if (!BridgeSystem.IsClientConnected || Main.gameMenu)
            {
                return;
            }
            var a = BridgeSystem.GetCurrentAction();
            if (a.HasAim)
            {
                Main.mouseX = (int)(a.AimX - Main.screenPosition.X);
                Main.mouseY = (int)(a.AimY - Main.screenPosition.Y);
            }
        }

        public override void PostUpdateEverything()
        {
            if (!Main.gameMenu)
            {
                Episode.Ticks++;
            }

            // Drain a bounded number per frame so a flood can't stall a frame.
            for (int i = 0; i < 16 && Jobs.TryDequeue(out var job); i++)
            {
                try
                {
                    job.Result = job.Work();
                }
                catch (Exception ex)
                {
                    job.Result = new JObject { ["ok"] = false, ["error"] = ex.GetType().Name + ": " + ex.Message };
                }
                job.Done.Set();
            }
        }

        // ---- commands (all run on the game thread) ----------------------

        public static JObject SetArena(JObject req)
        {
            var p = Main.LocalPlayer;
            float x = (float?)req["x"] ?? p.position.X;
            float y = (float?)req["y"] ?? p.position.Y;
            Episode.ArenaPosition = new Vector2(x, y);
            return new JObject { ["ok"] = true, ["arena"] = new JObject { ["x"] = x, ["y"] = y } };
        }

        /// <summary>
        /// Clean slate for a new episode: despawn every hostile NPC (incl.
        /// all Moon Lord parts) and every hostile projectile, heal to full,
        /// clear debuffs, move the player to the arena position (if set),
        /// zero velocity, reset counters. Does NOT respawn a dead player
        /// (the harness waits for the game's own respawn first) and does
        /// not resummon minions (the harness does that with select_slot +
        /// use, as a logged setup step).
        /// </summary>
        public static JObject ResetEpisode(JObject req)
        {
            var p = Main.LocalPlayer;
            if (p.dead)
            {
                return new JObject { ["ok"] = false, ["error"] = "player is dead; wait for respawn" };
            }

            int npcsRemoved = 0;
            foreach (var npc in Main.npc)
            {
                if (npc != null && npc.active && !npc.friendly && !npc.townNPC)
                {
                    npc.active = false;
                    npc.life = 0;
                    npcsRemoved++;
                }
            }

            int projRemoved = 0;
            foreach (var proj in Main.projectile)
            {
                if (proj != null && proj.active && proj.hostile)
                {
                    proj.Kill();
                    projRemoved++;
                }
            }

            p.statLife = p.statLifeMax2;
            p.statMana = p.statManaMax2;
            for (int b = p.buffType.Length - 1; b >= 0; b--)
            {
                int t = p.buffType[b];
                if (t > 0 && Main.debuff[t])
                {
                    p.DelBuff(b);
                }
            }

            bool moved = false;
            if (Episode.ArenaPosition.HasValue)
            {
                p.position = Episode.ArenaPosition.Value;
                p.velocity = Vector2.Zero;
                p.fallStart = (int)(p.position.Y / 16f);
                moved = true;
            }

            // Gravestones from earlier deaths pile up around the arena: clear
            // them in a generous box around the arena mark (or the player).
            var c0 = Episode.ArenaPosition ?? p.position;
            int tx = (int)(c0.X / 16f), ty = (int)(c0.Y / 16f);
            int graves = Setup.ClearTombstones(tx - 600, ty - 300, tx + 600, ty + 100);

            BridgeSystem.ReleaseAll();
            Episode.Reset();
            return new JObject
            {
                ["ok"] = true,
                ["npcs_removed"] = npcsRemoved,
                ["projectiles_removed"] = projRemoved,
                ["moved_to_arena"] = moved,
                ["tombstone_tiles_removed"] = graves,
            };
        }

        /// <summary>spawn_npc {type}: NPC.SpawnOnPlayer, the same call boss
        /// summon items use. Default type = Moon Lord core (spawns the whole
        /// boss).</summary>
        public static JObject SpawnNpc(JObject req)
        {
            int type = (int?)req["type"] ?? NPCID.MoonLordCore;
            NPC.SpawnOnPlayer(Main.myPlayer, type);
            return new JObject { ["ok"] = true, ["type"] = type };
        }

        /// <summary>spawn_projectile {type, x, y, vx, vy, damage, knockback}:
        /// for the dodge gym. Positions are world pixels; if dx/dy are given
        /// instead of x/y they are offsets from the player's centre. The
        /// projectile is forced hostile so it can hit the player.</summary>
        public static JObject SpawnProjectile(JObject req)
        {
            var p = Main.LocalPlayer;
            int type = (int?)req["type"] ?? ProjectileID.PhantasmalBolt;
            float x = req["x"] != null ? (float)req["x"] : p.Center.X + ((float?)req["dx"] ?? 0f);
            float y = req["y"] != null ? (float)req["y"] : p.Center.Y + ((float?)req["dy"] ?? 0f);
            float vx = (float?)req["vx"] ?? 0f;
            float vy = (float?)req["vy"] ?? 0f;
            int damage = (int?)req["damage"] ?? 60;
            float kb = (float?)req["knockback"] ?? 0f;

            int idx = Projectile.NewProjectile(new EntitySource_Misc("FruitFlyBridge.DodgeGym"),
                x, y, vx, vy, type, damage, kb, Main.myPlayer);
            if (idx >= 0 && idx < Main.maxProjectiles)
            {
                var proj = Main.projectile[idx];
                proj.hostile = true;
                proj.friendly = false;
            }
            return new JObject { ["ok"] = true, ["index"] = idx, ["type"] = type };
        }

        /// <summary>quick_heal: the game's own Quick Heal (the H key) --
        /// drinks the best healing potion in the inventory, if potion
        /// sickness allows. Reports HP before/after.</summary>
        public static JObject QuickHeal(JObject req)
        {
            var p = Main.LocalPlayer;
            int before = p.statLife;
            bool sick = p.potionDelay > 0;
            if (!sick && !p.dead)
            {
                p.QuickHeal();
            }
            return new JObject { ["ok"] = true, ["hp_before"] = before, ["hp_after"] = p.statLife,
                                 ["potion_sick"] = sick, ["healed"] = p.statLife > before };
        }

        /// <summary>select_slot {slot}: hotbar slot 0-9, so the harness can
        /// cast summon weapons during setup.</summary>
        public static JObject SelectSlot(JObject req)
        {
            int slot = (int?)req["slot"] ?? 0;
            if (slot < 0 || slot > 9)
            {
                return new JObject { ["ok"] = false, ["error"] = "slot must be 0-9" };
            }
            Main.LocalPlayer.selectedItem = slot;
            var item = Main.LocalPlayer.inventory[slot];
            return new JObject { ["ok"] = true, ["slot"] = slot, ["item_type"] = item.type, ["item_name"] = item.Name };
        }
    }

    /// <summary>
    /// Character and arena setup, so neither has to be downloaded: the
    /// loadout lives in training/moonlord/loadout.json and is applied to
    /// whatever character is loaded. Names are Terraria's internal ID names
    /// (ItemID / BuffID constants, e.g. "StardustDragonStaff"); unknown
    /// names are reported back, never guessed.
    /// </summary>
    public static class Setup
    {
        /// <summary>
        /// An item reference from loadout.json: an internal name
        /// ("RainbowWhip"), a numeric item id (4914), or a list of
        /// alternatives tried in order (["RainbowWhip", 4914]). Returns 0
        /// for "empty", -1 if nothing in it exists in this tModLoader build.
        /// </summary>
        public static int ResolveItemId(JToken tok)
        {
            if (tok == null || tok.Type == JTokenType.Null)
            {
                return 0;
            }
            if (tok.Type == JTokenType.Integer)
            {
                int id = (int)tok;
                return id > 0 && id < ItemLoader.ItemCount ? id : -1;
            }
            if (tok.Type == JTokenType.String)
            {
                var name = (string)tok;
                return ItemID.Search.ContainsName(name) ? ItemID.Search.GetId(name) : -1;
            }
            if (tok is JArray alts)
            {
                foreach (var alt in alts)
                {
                    int id = ResolveItemId(alt);
                    if (id > 0)
                    {
                        return id;
                    }
                }
                return -1;
            }
            return -1;
        }

        private static Item MakeItem(JToken tok, int stack, JArray unknown)
        {
            var item = new Item();
            int id = ResolveItemId(tok);
            if (id == 0)
            {
                return item; // empty slot
            }
            if (id < 0)
            {
                unknown.Add(tok.ToString(Newtonsoft.Json.Formatting.None));
                return item;
            }
            item.SetDefaults(id);
            item.stack = Math.Max(1, Math.Min(stack, item.maxStack));
            return item;
        }

        /// <summary>
        /// equip_loadout {armor: [head, body, legs], accessories: [...],
        /// hotbar: [name or null, ...] (slots 0-9), inventory: [{name, stack}],
        /// max_stats: true}. Overwrites those slots on the loaded character
        /// (saved with it when the game saves). max_stats = all Life
        /// Crystals, Life Fruit and Mana Crystals consumed, plus the Demon
        /// Heart accessory slot (only usable in Expert/Master worlds).
        /// </summary>
        public static JObject EquipLoadout(JObject req)
        {
            var p = Main.LocalPlayer;
            var unknown = new JArray();

            if (req["armor"] is JArray armor)
            {
                for (int i = 0; i < Math.Min(3, armor.Count); i++)
                {
                    p.armor[i] = MakeItem(armor[i], 1, unknown);
                }
            }
            if (req["accessories"] is JArray acc)
            {
                for (int i = 0; i < 7; i++)
                {
                    p.armor[3 + i] = MakeItem(i < acc.Count ? acc[i] : null, 1, unknown);
                }
            }
            if (req["hotbar"] is JArray hotbar)
            {
                for (int i = 0; i < Math.Min(10, hotbar.Count); i++)
                {
                    p.inventory[i] = MakeItem(hotbar[i], 1, unknown);
                }
            }
            if (req["inventory"] is JArray inv)
            {
                for (int i = 0; i < inv.Count && 10 + i < 50; i++)
                {
                    var entry = inv[i] as JObject;
                    p.inventory[10 + i] = MakeItem(entry?["name"], (int?)entry?["stack"] ?? 1, unknown);
                }
            }
            if ((bool?)req["max_stats"] ?? false)
            {
                p.ConsumedLifeCrystals = Player.LifeCrystalMax;
                p.ConsumedLifeFruit = Player.LifeFruitMax;
                p.ConsumedManaCrystals = Player.ManaCrystalMax;
                p.extraAccessory = true;
                p.statLifeMax = 100 + Player.LifeCrystalMax * 20 + Player.LifeFruitMax * 5;
                p.statManaMax = 20 + Player.ManaCrystalMax * 20;
                p.statLife = p.statLifeMax;
                p.statMana = p.statManaMax;
            }
            return new JObject { ["ok"] = unknown.Count == 0, ["unknown_items"] = unknown,
                                 ["error"] = unknown.Count == 0 ? null : "some item names were not found (slots left empty)" };
        }

        /// <summary>apply_buffs {names: [...], seconds}: e.g. potions and the
        /// arena-furniture buffs (Campfire, HeartLamp...), so the arena needs
        /// no furniture.</summary>
        public static JObject ApplyBuffs(JObject req)
        {
            var p = Main.LocalPlayer;
            int ticks = (int)(((float?)req["seconds"] ?? 600f) * 60f);
            var unknown = new JArray();
            int applied = 0;
            if (req["names"] is JArray names)
            {
                foreach (var n in names)
                {
                    var name = (string)n;
                    if (!BuffID.Search.ContainsName(name))
                    {
                        unknown.Add(name);
                        continue;
                    }
                    p.AddBuff(BuffID.Search.GetId(name), ticks);
                    applied++;
                }
            }
            return new JObject { ["ok"] = unknown.Count == 0, ["applied"] = applied, ["unknown_buffs"] = unknown,
                                 ["error"] = unknown.Count == 0 ? null : "some buff names were not found" };
        }

        /// <summary>
        /// build_arena: rows of planter boxes centred on the player (the
        /// first row_spacing_tiles above their feet), campfires standing on
        /// the lowest row, and heart lanterns hanging from a solid block set
        /// into the second row. Every furniture_spacing_tiles across.
        /// Items are loadout references (name / id / alternatives); their
        /// own createTile + placeStyle decide what gets placed, so no tile
        /// style numbers are hard-coded. Only fills empty tiles, so it can
        /// be re-run. Marks the arena at the player's position.
        /// </summary>
        public static JObject BuildArena(JObject req)
        {
            var p = Main.LocalPlayer;
            int half = (int?)req["half_width_tiles"] ?? 100;
            int rows = (int?)req["rows"] ?? 4;
            int spacing = (int?)req["row_spacing_tiles"] ?? 12;
            int furnSpacing = (int?)req["furniture_spacing_tiles"] ?? 60;
            var unknown = new JArray();
            var floor = Template(req["floor_item"] ?? (JToken)"DayBloomPlanterBox", unknown);
            var fire = Template(req["campfire_item"] ?? (JToken)"Campfire", unknown);
            var lantern = Template(req["lantern_item"] ?? (JToken)"HeartLantern", unknown);
            var block = Template(req["hanger_block_item"] ?? (JToken)"GrayBrick", unknown);
            if (unknown.Count > 0)
            {
                return new JObject { ["ok"] = false, ["error"] = "unknown arena items", ["unknown_items"] = unknown };
            }

            // Build around an explicit spot (e.g. the existing arena mark) if
            // given, else around the player. x/y = player top-left, as in set_arena.
            var origin = new Vector2((float?)req["x"] ?? p.position.X, (float?)req["y"] ?? p.position.Y);
            bool replacePlatforms = (bool?)req["replace_platforms"] ?? true;
            int cx = (int)((origin.X + p.width / 2f) / 16f);
            int feet = (int)((origin.Y + p.height) / 16f);
            int RowY(int r) => feet - r * spacing;
            bool InWorld(int x, int y) => x > 10 && y > 10 && x < Main.maxTilesX - 10 && y < Main.maxTilesY - 10;

            // clear: wipe everything (sky islands, trees, old builds, liquid,
            // walls) in the arena box first, from just above the floor line up
            // to 3 tiles past the top row and 2 past each side.
            int clearedTiles = 0;
            if ((bool?)req["clear"] ?? false)
            {
                clearedTiles = ClearArea(cx - half - 2, RowY(rows) - 3, cx + half + 2, feet - 1);
            }

            int floorPlaced = 0;
            for (int r = 1; r <= rows; r++)
            {
                for (int tx = cx - half; tx <= cx + half; tx++)
                {
                    if (!InWorld(tx, RowY(r)))
                    {
                        continue;
                    }
                    // Swap out the old wooden-platform rows from the first arena version.
                    if (replacePlatforms && Has(tx, RowY(r), TileID.Platforms))
                    {
                        WorldGen.KillTile(tx, RowY(r), false, false, true);
                    }
                    if (!Main.tile[tx, RowY(r)].HasTile
                        && WorldGen.PlaceTile(tx, RowY(r), floor.createTile, true, false, -1, floor.placeStyle))
                    {
                        floorPlaced++;
                    }
                }
            }

            // Furniture every `furnRowEvery` rows so a tall arena stays inside
            // the ~170x125-tile buff range: campfires on rows 1, 1+N, 1+2N...,
            // heart lanterns hung from a block set into rows 3, 3+N, ...
            // (or row 2/1 in short arenas).
            int furnRowEvery = Math.Max(1, (int?)req["furniture_row_every"] ?? 5);
            int lanternOffset = Math.Min(2, rows - 1);
            var fires = new JArray();
            var lanterns = new JArray();
            for (int r = 1; r <= rows; r++)
            {
                bool fireRow = (r - 1) % furnRowEvery == 0;
                bool lanternRow = r - 1 - lanternOffset >= 0 && (r - 1 - lanternOffset) % furnRowEvery == 0;
                if (!fireRow && !lanternRow)
                {
                    continue;
                }
                for (int k = -half / furnSpacing; k <= half / furnSpacing; k++)
                {
                    if (fireRow)
                    {
                        // Campfire: 3x2, origin bottom-middle, standing on row r.
                        int fx = cx + k * furnSpacing;
                        int fy = RowY(r) - 1;
                        if (InWorld(fx, fy))
                        {
                            bool ok = Has(fx, fy, fire.createTile) || WorldGen.PlaceObject(fx, fy, fire.createTile, true, fire.placeStyle);
                            if (!ok)
                            {
                                // Fallback if it won't stand on planter boxes:
                                // a 3-tile solid pad under it, then retry.
                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    Replace(fx + dx, RowY(r), block);
                                }
                                ok = WorldGen.PlaceObject(fx, fy, fire.createTile, true, fire.placeStyle);
                            }
                            fires.Add(new JObject { ["row"] = r, ["tx"] = fx, ["ty"] = fy, ["ok"] = ok && Has(fx, fy, fire.createTile) });
                        }
                    }
                    if (lanternRow)
                    {
                        // Heart lantern: 1x2, hangs from the bottom of a solid
                        // block set into row r, half a spacing from the campfires.
                        int lx = cx + k * furnSpacing + furnSpacing / 2;
                        int ly = RowY(r);
                        if (lx <= cx + half && InWorld(lx, ly + 1))
                        {
                            Replace(lx, ly, block);
                            bool ok = Has(lx, ly + 1, lantern.createTile)
                                      || WorldGen.PlaceObject(lx, ly + 1, lantern.createTile, true, lantern.placeStyle);
                            lanterns.Add(new JObject { ["row"] = r, ["tx"] = lx, ["ty"] = ly + 1, ["ok"] = ok && Has(lx, ly + 1, lantern.createTile) });
                        }
                    }
                }
            }

            // Rows above the full-gravity line (0 = the whole arena has normal gravity).
            int RowsInLowGravity()
            {
                int n = 0;
                for (int r = 1; r <= rows; r++)
                    if (RowY(r) * 16.0 < Episode.FullGravityY()) n++;
                return n;
            }

            Episode.ArenaPosition = origin;
            bool allOk = true;
            foreach (var f in fires) allOk &= (bool)f["ok"];
            foreach (var l in lanterns) allOk &= (bool)l["ok"];
            return new JObject
            {
                ["ok"] = allOk,
                ["error"] = allOk ? null : "some furniture could not be placed (see campfires/lanterns)",
                ["cleared_tiles"] = clearedTiles,
                ["floor_tiles_placed"] = floorPlaced,
                ["campfires"] = fires,
                ["lanterns"] = lanterns,
                ["arena"] = new JObject { ["x"] = origin.X, ["y"] = origin.Y },
                ["top_row_y"] = RowY(rows) * 16.0,
                ["full_gravity_y"] = Episode.FullGravityY(),
                ["rows_in_low_gravity"] = RowsInLowGravity(),
            };
        }

        /// <summary>
        /// Wipe a rectangle of tiles (inclusive): blocks, walls, liquid, wires,
        /// plus the chests and signs stored there. Used to clear sky islands
        /// and anything else out of the arena. Direct clearing, no item drops.
        /// </summary>
        public static int ClearArea(int x0, int y0, int x1, int y1)
        {
            x0 = Math.Max(x0, 5); y0 = Math.Max(y0, 5);
            x1 = Math.Min(x1, Main.maxTilesX - 6); y1 = Math.Min(y1, Main.maxTilesY - 6);
            for (int i = 0; i < Main.maxChests; i++)
            {
                var c = Main.chest[i];
                if (c != null && c.x >= x0 && c.x <= x1 && c.y >= y0 && c.y <= y1)
                {
                    Main.chest[i] = null;
                }
            }
            for (int i = 0; i < Main.sign.Length; i++)
            {
                var sg = Main.sign[i];
                if (sg != null && sg.x >= x0 && sg.x <= x1 && sg.y >= y0 && sg.y <= y1)
                {
                    Main.sign[i] = null;
                }
            }
            int cleared = 0;
            for (int x = x0; x <= x1; x++)
            {
                for (int y = y0; y <= y1; y++)
                {
                    var t = Main.tile[x, y];
                    if (t.HasTile || t.WallType != 0 || t.LiquidAmount != 0)
                    {
                        cleared++;
                    }
                    t.ClearEverything();
                }
            }
            WorldGen.RangeFrame(x0, y0, x1, y1);
            return cleared;
        }

        /// <summary>
        /// Remove gravestones (TileID.Tombstones) in a rectangle, or the whole
        /// world if no rectangle is given, with their epitaph signs. Called at
        /// every reset_episode around the arena, and by clear_tombstones.
        /// </summary>
        public static int ClearTombstones(int x0 = 0, int y0 = 0, int x1 = -1, int y1 = -1)
        {
            if (x1 < 0) { x1 = Main.maxTilesX - 1; }
            if (y1 < 0) { y1 = Main.maxTilesY - 1; }
            x0 = Math.Max(x0, 0); y0 = Math.Max(y0, 0);
            x1 = Math.Min(x1, Main.maxTilesX - 1); y1 = Math.Min(y1, Main.maxTilesY - 1);
            int removed = 0;
            for (int x = x0; x <= x1; x++)
            {
                for (int y = y0; y <= y1; y++)
                {
                    var t = Main.tile[x, y];
                    if (t.HasTile && t.TileType == TileID.Tombstones)
                    {
                        Sign.KillSign(x, y);
                        t.ClearTile();
                        removed++;
                    }
                }
            }
            if (removed > 0)
            {
                WorldGen.RangeFrame(x0, y0, x1, y1);
            }
            return removed;
        }

        public static JObject ClearTombstonesCommand(JObject req)
        {
            int n = ClearTombstones();
            return new JObject { ["ok"] = true, ["tombstone_tiles_removed"] = n };
        }

        private static Item Template(JToken tok, JArray unknown)
        {
            var item = MakeItem(tok, 1, unknown);
            if (item.type > 0 && item.createTile < 0)
            {
                unknown.Add(tok.ToString(Newtonsoft.Json.Formatting.None) + " (does not place a tile)");
            }
            return item;
        }

        private static bool Has(int x, int y, int tileType)
        {
            var t = Main.tile[x, y];
            return t.HasTile && t.TileType == tileType;
        }

        /// <summary>Make (x, y) the given block, removing whatever 1x1 tile is there.</summary>
        private static void Replace(int x, int y, Item block)
        {
            if (Has(x, y, block.createTile))
            {
                return;
            }
            if (Main.tile[x, y].HasTile)
            {
                WorldGen.KillTile(x, y, false, false, true);
            }
            WorldGen.PlaceTile(x, y, block.createTile, true, false, -1, block.placeStyle);
        }
    }

    public class HarnessGlobalNPC : GlobalNPC
    {
        public override void OnKill(NPC npc)
        {
            if (npc.type == NPCID.MoonLordCore && !Episode.BossKilled)
            {
                Episode.BossKilled = true;
                Episode.BossKillTick = Episode.Ticks;
            }
        }
    }
}
