using Newtonsoft.Json.Linq;
using Terraria;
using Terraria.ID;

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

        // Player centre is included too (Goal 2), since projectile and boss
        // positions below are centres.
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
                    ["cx"] = player.Center.X,
                    ["cy"] = player.Center.Y,
                    ["direction"] = player.direction,
                    ["hp"] = player.statLife,
                    ["max_hp"] = player.statLifeMax2, // with buffs/accessories (was statLifeMax)
                    ["mana"] = player.statMana,
                    ["max_mana"] = player.statManaMax,
                    ["on_ground"] = player.velocity.Y == 0f,
                    // Vanilla Player.Update branches on these two fields
                    // before any control processing, so while either is
                    // true no input (ours or the keyboard's) does anything.
                    // Added after a live test where the character died to
                    // a zombie swarm and the bot kept acting on the corpse.
                    ["dead"] = player.dead,
                    ["ghost"] = player.ghost,
                    // Goal 2 (Moon Lord): motion, flight and summoner state.
                    ["vx"] = player.velocity.X,
                    ["vy"] = player.velocity.Y,
                    ["width"] = player.width,
                    ["height"] = player.height,
                    ["wing_time"] = player.wingTime,
                    // Actual gravity this tick (0.4 normally; lower near space, see Episode.FullGravityY).
                    ["gravity"] = player.gravity,
                    ["wing_time_max"] = player.wingTimeMax,
                    ["mount_active"] = player.mount != null && player.mount.Active,
                    ["immune"] = player.immune,
                    ["minions"] = player.numMinions,
                    ["max_minions"] = player.maxMinions,
                    ["selected_item"] = player.selectedItem,
                    // Healing: potion sickness ticks left, and healing potions carried.
                    ["potion_delay"] = player.potionDelay,
                    ["heal_potions"] = CountHealPotions(player),
                },
                ["projectiles"] = BuildHostileProjectileArray(player),
                ["episode"] = Episode.ToJson(),
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
                    ["game_is_active"] = Main.instance?.IsActive ?? false,
                    ["game_paused"] = Main.gamePaused,
                    ["control_left"] = player.controlLeft,
                    ["control_right"] = player.controlRight,
                    ["control_jump"] = player.controlJump,
                    ["control_up"] = player.controlUp,
                    ["control_down"] = player.controlDown,
                    ["velocity_x"] = player.velocity.X,
                    ["is_operating_another_entity"] = player.isOperatingAnotherEntity,
                    ["mount_active"] = player.mount != null && player.mount.Active,
                    ["frozen"] = player.frozen,
                    ["stoned"] = player.stoned,
                    ["cc_immune"] = player.CCed,
                    // Item-use diagnostics (whip assist never swung in the
                    // first live test; these show which gate blocks it).
                    ["control_use_item"] = player.controlUseItem,
                    ["release_use_item"] = player.releaseUseItem,
                    ["delay_use_item"] = player.delayUseItem,
                    ["item_animation"] = player.itemAnimation,
                    ["mouse_interface"] = player.mouseInterface,
                    ["held_item_type"] = player.HeldItem?.type ?? 0,
                    ["held_item_name"] = player.HeldItem?.Name ?? "",
                    ["mouse_world_x"] = Main.MouseWorld.X,
                    ["mouse_world_y"] = Main.MouseWorld.Y,
                    ["whip_projectiles"] = CountOwnWhips(player),
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

        // Live projectiles owned by the player that are whips (the swing
        // itself is a projectile) -- direct evidence a swing happened.
        private static int CountOwnWhips(Player player)
        {
            int n = 0;
            foreach (var proj in Main.projectile)
            {
                if (proj != null && proj.active && proj.owner == player.whoAmI && ProjectileID.Sets.IsAWhip[proj.type])
                {
                    n++;
                }
            }
            return n;
        }

        private static int CountHealPotions(Player player)
        {
            int n = 0;
            for (int i = 0; i < 58; i++)
            {
                var it = player.inventory[i];
                if (it != null && it.stack > 0 && it.potion && it.healLife > 0)
                {
                    n += it.stack;
                }
            }
            return n;
        }

        private const int ProjectileRadius = 1600; // pixels (~100 tiles)

        // Goal 2: hostile projectiles (what the fly has to dodge). Centre
        // position, velocity and hitbox size, so the Python side can work
        // out where each one is heading and how soon it arrives.
        private static JArray BuildHostileProjectileArray(Player player)
        {
            var list = new JArray();
            foreach (var proj in Main.projectile)
            {
                if (proj == null || !proj.active || !proj.hostile)
                {
                    continue;
                }

                var dx = proj.Center.X - player.Center.X;
                var dy = proj.Center.Y - player.Center.Y;
                if (dx * dx + dy * dy > ProjectileRadius * ProjectileRadius)
                {
                    continue;
                }

                list.Add(new JObject
                {
                    ["type"] = proj.type,
                    ["cx"] = proj.Center.X,
                    ["cy"] = proj.Center.Y,
                    ["vx"] = proj.velocity.X,
                    ["vy"] = proj.velocity.Y,
                    ["width"] = proj.width,
                    ["height"] = proj.height,
                    ["damage"] = proj.damage,
                });
            }
            return list;
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
                // Bosses (and every Moon Lord part) are always reported:
                // the Moon Lord is far bigger than the 800px radius.
                bool always = npc.boss || Episode.IsMoonLordPart(npc.type);
                if (!always && dx * dx + dy * dy > NpcRadius * NpcRadius)
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
                    // Goal 2 additions. x/y above stay the top-left corner
                    // (unchanged for the existing encoder); cx/cy = centre.
                    ["cx"] = npc.Center.X,
                    ["cy"] = npc.Center.Y,
                    ["vx"] = npc.velocity.X,
                    ["vy"] = npc.velocity.Y,
                    ["width"] = npc.width,
                    ["height"] = npc.height,
                    ["max_hp"] = npc.lifeMax,
                    ["boss"] = npc.boss,
                    ["moon_lord_part"] = Episode.IsMoonLordPart(npc.type),
                    ["dont_take_damage"] = npc.dontTakeDamage,
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
