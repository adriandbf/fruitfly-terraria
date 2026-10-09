using Terraria;
using Terraria.DataStructures;
using Terraria.GameInput;
using Terraria.ModLoader;

namespace FruitFlyBridge
{
    /// <summary>
    /// Applies the bridge's last-requested action to the LOCAL player every
    /// tick, so keys stay "held" between decisions rather than needing to be
    /// re-sent at 60 Hz. Only touches the client's own player (whoAmI ==
    /// Main.myPlayer); other players are never affected.
    ///
    /// Sets Player.controlX directly. Confirmed by decompiling the actual
    /// tModLoader v2026.07.3.0 build (Terraria.GameInput.TriggersSet.CopyInto):
    /// the game copies its own trigger state onto p.controlLeft/Right/Jump
    /// (and controlUseItem, from MouseLeft) FIRST, and calls
    /// PlayerLoader.ProcessTriggers(p, this) — our hook — as the LAST line of
    /// that same method. So by the time this runs, controlX already holds
    /// the real keyboard state; writing controlX here is what overrides it,
    /// and writing the triggersSet parameter instead (tried previously) is a
    /// no-op for movement, since it's already been read.
    ///
    /// Only overrides while BridgeSystem.IsClientConnected — otherwise this
    /// would force every control to "released" the instant the mod loads,
    /// even with no bridge client running, which is what blocked ALL
    /// movement (including the real keyboard) on an earlier test.
    /// </summary>
    public class BridgePlayer : ModPlayer
    {
        public override void ProcessTriggers(TriggersSet triggersSet)
        {
            if (Player.whoAmI != Main.myPlayer || !BridgeSystem.IsClientConnected)
            {
                return;
            }

            Apply();
        }

        /// <summary>
        /// Goal 2: also applied from SetControls, which tModLoader calls a
        /// little later in Player.Update (after the double-tap timers) and,
        /// per its docs, even in text-entry mode where ProcessTriggers is
        /// skipped. Applying twice per tick is harmless.
        /// </summary>
        public override void SetControls()
        {
            if (Player.whoAmI != Main.myPlayer || !BridgeSystem.IsClientConnected)
            {
                return;
            }

            Apply();
        }

        private void Apply()
        {
            var action = BridgeSystem.GetCurrentAction();
            if (Episode.PassthroughMovement)
            {
                // Human demo: leave the real keyboard alone; only add item use
                // (whip assist) when the bridge asks for it.
                if (action.UseItem)
                {
                    Player.controlUseItem = true;
                    Player.delayUseItem = false;
                    Player.releaseUseItem = true;
                    Player.mouseInterface = false;
                }
                return;
            }
            Player.controlLeft = action.Left;
            Player.controlRight = action.Right;
            Player.controlJump = action.Jump;
            Player.controlUseItem = action.UseItem;
            if (action.UseItem)
            {
                // Holding use from a script never produces the release/press
                // edge vanilla expects, and any stale gate would then block
                // every swing (first live whip test: assist held `use` for
                // 178 ticks, no swing). Clear the gates so a held `use`
                // behaves like autoswing: reuse as soon as the animation ends.
                Player.delayUseItem = false;
                Player.releaseUseItem = true;
                Player.mouseInterface = false;
            }
            // Up/down: wings hover/ascend, flying mounts, platforms (Goal 2).
            Player.controlUp = action.Up;
            Player.controlDown = action.Down;
        }

        // Episode counters for the Moon Lord harness (Harness.cs).
        public override void OnHurt(Player.HurtInfo info)
        {
            if (Player.whoAmI == Main.myPlayer)
            {
                Episode.Hits++;
                Episode.DamageTaken += info.Damage;
            }
        }

        public override void Kill(double damage, int hitDirection, bool pvp, PlayerDeathReason damageSource)
        {
            if (Player.whoAmI == Main.myPlayer)
            {
                Episode.Deaths++;
            }
        }
    }
}
