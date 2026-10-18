"""Webserver: MJPEG-stream, WebSocket met herkenningen en de gebouwde frontend.

Gebruik (vanuit de projectmap):
    python -m backend.main                          # webcam 0, http://127.0.0.1:8000
    python -m backend.main --bron 1                 # andere camera
    python -m backend.main --bron data/voorbeelden/foto.jpg   # testen zonder webcam
    python -m backend.main --stabiel 8 --min-kans 0.6

Endpoints:
    GET /video    MJPEG-stream van de camera
    WS  /ws       JSON bij elke kaartwissel ("type": "change") en elke 100 ms ("type": "update")
    GET /status   laatste status als JSON
    GET /test     eenvoudige testpagina (zonder frontend-build)
    GET /         de frontend uit frontend/dist (na `npm run build`)
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.herkenner import Herkenner, Instellingen
from tarot.config import ROOT

FRONTEND_DIST = ROOT / "frontend" / "dist"
TESTPAGINA = Path(__file__).parent / "test.html"
UPDATE_INTERVAL = 0.1  # seconden tussen WebSocket-updates


def maak_app(inst: Instellingen) -> FastAPI:
    herkenner = Herkenner(inst)

    @asynccontextmanager
    async def levensduur(_app):
        herkenner.start()
        yield
        herkenner.stop()

    app = FastAPI(title="Tarot herkenner", lifespan=levensduur)

    @app.get("/video")
    async def video():
        async def frames():
            vorige = -1
            while True:
                jpeg, nr = herkenner.laatste_jpeg()
                if jpeg is not None and nr != vorige:
                    vorige = nr
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.01)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/status")
    async def status():
        return herkenner.status()[0]

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        vorige_versie = None
        try:
            while True:
                status, versie = herkenner.status()
                status["type"] = "change" if versie != vorige_versie else "update"
                vorige_versie = versie
                await socket.send_json(status)
                await asyncio.sleep(UPDATE_INTERVAL)
        except WebSocketDisconnect:
            pass

    @app.get("/test")
    async def test():
        return FileResponse(TESTPAGINA)

    # Als laatste: alles wat geen route hierboven is, komt uit de frontend-build.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:
        print(f"Let op: {FRONTEND_DIST} bestaat niet; alleen /test, /video en /ws werken.")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bron", default="0", help="camera-index, of pad naar video/foto")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = ook bereikbaar vanaf andere apparaten")
    parser.add_argument("--poort", type=int, default=8000)
    parser.add_argument("--stabiel", type=int, default=5, help="frames op rij voor een nieuwe kaart")
    parser.add_argument("--weg", type=int, default=5, help="frames zonder kaart voor 'geen kaart'")
    parser.add_argument("--min-conf", type=float, default=0.5, help="minimale detector-confidence")
    parser.add_argument("--min-kans", type=float, default=0.5, help="minimale classifier-kans")
    args = parser.parse_args()

    inst = Instellingen(bron=args.bron, stabiel_n=args.stabiel, weg_n=args.weg,
                        min_conf=args.min_conf, min_kans=args.min_kans)
    uvicorn.run(maak_app(inst), host=args.host, port=args.poort)


if __name__ == "__main__":
    main()
