import type { Status } from "./types";

/** Tekent de omtrek van de gedetecteerde kaart over de video. */
export class Overlay {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private laatste: Status | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    new ResizeObserver(() => this.laatste && this.teken(this.laatste)).observe(canvas);
  }

  teken(status: Status): void {
    this.laatste = status;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const b = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(b * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(b * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, b, h);
    if (!status.corners || status.frame_size[0] === 0) return;

    // Framepixels -> weergavepixels (de video vult het canvas in de breedte).
    const schaal = b / status.frame_size[0];
    const punten = status.corners.map(([x, y]) => [x * schaal, y * schaal] as const);

    const herkend = status.card !== null;
    ctx.lineWidth = 3;
    ctx.strokeStyle = herkend ? "#e8c66a" : "rgba(255, 255, 255, 0.7)";
    ctx.beginPath();
    punten.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.stroke();

    if (herkend && status.label) {
      const [x, y] = punten[0];
      ctx.font = "600 16px system-ui, sans-serif";
      const tekst = status.reversed ? `${status.label} (omgekeerd)` : status.label;
      const breedte = ctx.measureText(tekst).width;
      const ty = Math.max(y - 8, 22);
      ctx.fillStyle = "rgba(20, 14, 30, 0.8)";
      ctx.fillRect(x - 4, ty - 18, breedte + 8, 24);
      ctx.fillStyle = "#f5e6b8";
      ctx.fillText(tekst, x, ty);
    }
  }
}
