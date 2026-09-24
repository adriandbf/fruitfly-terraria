using Terraria;
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

            var action = BridgeSystem.GetCurrentAction();
            Player.controlLeft = action.Left;
            Player.controlRight = action.Right;
            Player.controlJump = action.Jump;
            Player.controlUseItem = action.UseItem;
        }
    }
}
