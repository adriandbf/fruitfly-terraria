using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json.Linq;
using Terraria;
using Terraria.ModLoader;

namespace FruitFlyBridge
{
    /// <summary>
    /// Holds the most recently requested action (which keys the fly wants held)
    /// so BridgePlayer can apply it every tick. One instance shared process-wide.
    /// </summary>
    public class ActionState
    {
        public bool Left, Right, Jump, UseItem;

        public void CopyFrom(ActionState other)
        {
            Left = other.Left;
            Right = other.Right;
            Jump = other.Jump;
            UseItem = other.UseItem;
        }
    }

    /// <summary>
    /// Local JSON-over-TCP bridge. Listens on 127.0.0.1:7778 only — this is a
    /// control channel for a process on the same machine, never exposed to the
    /// network. Protocol is newline-delimited JSON, one request/response per
    /// line, see bridge/README.md for the schema.
    ///
    /// </summary>
    public class BridgeSystem : ModSystem
    {
        public const int Port = 7778;

        private static readonly object ActionLock = new object();
        private static readonly ActionState PendingAction = new ActionState();
        private static int _connectedClients;

        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile bool _running;

        /// <summary>
        /// Whether a client is currently connected. BridgePlayer only overrides
        /// real input while this is true — otherwise the mod would force every
        /// control to "released" as soon as it loads (PendingAction defaults to
        /// all-false), blocking normal play even with no bridge client running.
        /// (First real test: the character wouldn't move at all, not even with
        /// the keyboard, once the mod was enabled — this is why.)
        /// </summary>
        public static bool IsClientConnected => _connectedClients > 0;

        /// <summary>Thread-safe snapshot of the most recent action request.</summary>
        public static ActionState GetCurrentAction()
        {
            var copy = new ActionState();
            lock (ActionLock)
            {
                copy.CopyFrom(PendingAction);
            }
            return copy;
        }

        private static void SetCurrentAction(ActionState next)
        {
            lock (ActionLock)
            {
                PendingAction.CopyFrom(next);
            }
        }

        public override void Load()
        {
            if (Main.dedServ)
            {
                return; // side=Client should already prevent this, belt and suspenders
            }

            _running = true;
            _listener = new TcpListener(IPAddress.Loopback, Port);
            _listener.Start();

            _acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "FruitFlyBridge-Accept" };
            _acceptThread.Start();

            Mod.Logger.Info($"FruitFlyBridge listening on 127.0.0.1:{Port}");
        }

        public override void Unload()
        {
            _running = false;
            try { _listener?.Stop(); } catch { /* already stopped */ }
        }

        private void AcceptLoop()
        {
            while (_running)
            {
                TcpClient client;
                try
                {
                    client = _listener.AcceptTcpClient();
                }
                catch (SocketException)
                {
                    break; // listener was stopped
                }

                var t = new Thread(() => HandleClient(client)) { IsBackground = true, Name = "FruitFlyBridge-Client" };
                t.Start();
            }
        }

        private void HandleClient(TcpClient client)
        {
            Interlocked.Increment(ref _connectedClients);
            try
            {
                using (client)
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream))
                using (var writer = new StreamWriter(stream) { AutoFlush = true })
                {
                    string line;
                    while (_running && (line = reader.ReadLine()) != null)
                    {
                        JObject response;
                        try
                        {
                            var request = JObject.Parse(line);
                            response = HandleRequest(request);
                        }
                        catch (Exception ex)
                        {
                            response = new JObject { ["ok"] = false, ["error"] = ex.Message };
                        }
                        writer.WriteLine(response.ToString(Newtonsoft.Json.Formatting.None));
                    }
                }
            }
            finally
            {
                Interlocked.Decrement(ref _connectedClients);
                SetCurrentAction(new ActionState()); // release all keys once the bridge disconnects
            }
        }

        private JObject HandleRequest(JObject request)
        {
            var cmd = (string)request["cmd"];
            switch (cmd)
            {
                case "ping":
                    return new JObject { ["ok"] = true, ["pong"] = true };

                case "get_state":
                    return GameState.BuildSnapshot();

                case "set_action":
                    var action = new ActionState
                    {
                        Left = (bool?)request["left"] ?? false,
                        Right = (bool?)request["right"] ?? false,
                        Jump = (bool?)request["jump"] ?? false,
                        UseItem = (bool?)request["use"] ?? false,
                    };
                    SetCurrentAction(action);
                    return new JObject { ["ok"] = true };

                default:
                    return new JObject { ["ok"] = false, ["error"] = $"unknown cmd '{cmd}'" };
            }
        }
    }
}
