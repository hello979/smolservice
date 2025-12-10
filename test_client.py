import asyncio
import websockets
import io
import time
import threading

# CONFIGURATION
# If testing Local:
URI = "ws://localhost:8000/ws/test_bot_01"
# If testing Render (Uncomment this and put your URL):
# URI = "wss://your-app-name.onrender.com/ws/test_bot_01"

async def bot_simulator():
    async with websockets.connect(URI) as websocket:
        print(f"\n✅ CONNECTED to {URI}")
        print("========================================")
        print("COMMANDS:")
        print(" Type any text -> Sends message to LLM")
        print(" Type 'int'    -> Simulates INTERRUPTION + Doubt")
        print(" Type 'exit'   -> Quits")
        print("========================================")

        # Background task to listen for incoming Audio/Text from Server
        listen_task = asyncio.create_task(receive_messages(websocket))

        # Main loop for User Input
        while True:
            user_input = await asyncio.to_thread(input, "\n📝 You (Client): ")

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            elif user_input.lower() == "int":
                # === SIMULATE INTERRUPTION SCENARIO ===
                print("\n🛑 TRIGGERING INTERRUPTION!")
                
                # 1. Send Interrupt Signal
                await websocket.send("__INTERRUPT__")
                print("   -> Sent '__INTERRUPT__'")
                
                # 2. Simulate the doubt immediately after
                doubt = "Wait, stop. Can you explain that in simple terms first?"
                print(f"   -> Sending Doubt: '{doubt}'")
                await websocket.send(doubt)

            else:
                # Normal Message
                await websocket.send(user_input)
                print("   -> Message sent.")

            # Keep connection alive
            await asyncio.sleep(0.1)

        listen_task.cancel()

async def receive_messages(websocket):
    """
    Listens for data from the server.
    Saves audio chunks to a file named 'streamed_output.wav'
    """
    chunk_count = 0
    # We open the file in Append Binary mode ('ab')
    # Note: Concatenating WAV headers creates 'clicks', 
    # but this verifies the data is arriving.
    with open("streamed_output.wav", "wb") as f:
        try:
            while True:
                message = await websocket.recv()

                if isinstance(message, bytes):
                    chunk_count += 1
                    file_size = len(message)
                    print(f"   🎧 Received Audio Chunk #{chunk_count} ({file_size} bytes)")
                    f.write(message)
                    f.flush()

                elif isinstance(message, str):
                    if message == "__CLEAR_LOCAL_BUFFER__":
                        print("\n   ⚡ COMMAND RECEIVED: CLEAR BUFFER (Stop playing audio)")
                        # In a real bot, you would run: audio_player.stop()
                    else:
                        print(f"   ℹ️ Server Message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print("\n❌ Connection Closed by Server")
        except Exception as e:
            print(f"Error receiving: {e}")

if __name__ == "__main__":
    # Clear previous test file
    open("streamed_output.wav", "wb").close()
    
    try:
        asyncio.run(bot_simulator())
    except KeyboardInterrupt:
        print("\nDisconnected.")