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
    /// Sets fields on the *triggersSet* parameter, not Player.controlX
    /// directly: ProcessTriggers runs before the game copies the current
    /// trigger state onto Player.controlX, so writing Player.controlX here
    /// gets immediately overwritten by the real (unpressed) keyboard state.
    /// (First build attempt set Player.controlX directly — compiled fine,
    /// did nothing in-game, this is the fix.)
    ///
    /// "UseItem" is a guess at the field name for the left-click/use-item
    /// trigger — it may actually be called MouseLeft. If this still doesn't
    /// make the character use the held item, that's the next thing to check.
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
            triggersSet.Left = action.Left;
            triggersSet.Right = action.Right;
            triggersSet.Jump = action.Jump;
            triggersSet.MouseLeft = action.UseItem;
        }
    }
}
