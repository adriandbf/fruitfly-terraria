using Terraria;
using Terraria.GameInput;
using Terraria.ModLoader;

namespace FruitFlyBridge
{
    /// <summary>
    /// Applies the bridge's last-requested action to the LOCAL player every
    /// tick, so keys stay "held" between decisions rather than needing to be
    /// re-sent at 60 Hz. Only touches Player.controlX fields on the client's
    /// own player (whoAmI == Main.myPlayer); other players are never affected.
    ///
    /// UNTESTED against a real tModLoader build — see BridgeSystem.cs.
    /// </summary>
    public class BridgePlayer : ModPlayer
    {
        public override void ProcessTriggers(TriggersSet triggersSet)
        {
            if (Player.whoAmI != Main.myPlayer)
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
