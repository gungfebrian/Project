# -*- coding: utf-8 -*-
"""
Server WebSocket yang berfungsi sebagai jembatan ke Robot Arm ESP32 via WiFi.
Versi ini dioptimalkan menggunakan aiohttp untuk komunikasi non-blocking,
menghasilkan performa yang jauh lebih cepat dan responsif.
"""
import asyncio
import websockets
import aiohttp # Menggantikan 'requests'
import re

# =================================================================
# --- KONFIGURASI PENTING: GANTI DENGAN ALAMAT IP ESP32 ANDA ---
# =================================================================
ESP32_IP = "192.168.1.150" # <-- GANTI INI DENGAN IP DARI ESP32 ANDA
# =================================================================

WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765

SERVO_COMMAND_PATTERN = re.compile(r"([BSEG])(\d+),")

# --- PERUBAHAN UTAMA: Fungsi handler koneksi ---
async def connection_handler(websocket, http_session):
    """Menerima perintah dari web dan meneruskannya ke ESP32 via HTTP GET secara asynchronous."""
    print(f"⚡️ Klien web terhubung dari {websocket.remote_address}")
    
    try:
        async for message in websocket:
            message = message.strip()
            if not message:
                continue

            print(f"> Menerima dari Web: '{message}'")
            
            match = SERVO_COMMAND_PATTERN.match(message)
            
            if match:
                servo_id = match.group(1)
                angle = match.group(2)
                url = f"http://{ESP32_IP}/move?servo={servo_id}&angle={angle}"
            else:
                action = message.replace(',', '')
                url = f"http://{ESP32_IP}/command?action={action}"
                
            try:
                print(f"   -> Mengirim permintaan ke: {url}")
                # Kirim permintaan ke ESP32 secara non-blocking
                # Timeout diatur di sini
                async with http_session.get(url, timeout=1.5) as response:
                    if response.status == 200:
                        print(f"   <- Respon ESP32: OK")
                    else:
                        print(f"   <- Peringatan: Respon ESP32 tidak OK (Status: {response.status})")
            except Exception as e:
                print(f"   ❌ Gagal terhubung ke ESP32 di {ESP32_IP}. Pastikan IP benar dan ESP32 online.")
                # print(f"   Error detail: {e}") # Uncomment untuk debugging

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Klien web terputus.")

async def main():
    """Fungsi utama untuk memulai server WebSocket dan aiohttp session."""
    print("🚀 Server Jembatan WiFi (Optimized) siap dijalankan.")
    print(f"   Meneruskan perintah ke ESP32 di alamat: http://{ESP32_IP}")
    
    # Membuat satu session aiohttp untuk digunakan kembali (best practice)
    async with aiohttp.ClientSession() as session:
        # Meneruskan session ke connection_handler
        handler = lambda ws: connection_handler(ws, session)
        
        async with websockets.serve(handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
            print(f"   Server WebSocket berjalan di ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
            print("   Buka file index.html di browser Anda.")
            await asyncio.Future()  # Berjalan selamanya

if __name__ == "__main__":
    print("Pastikan Anda sudah menjalankan 'pip3 install websockets aiohttp'")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server dihentikan.")
