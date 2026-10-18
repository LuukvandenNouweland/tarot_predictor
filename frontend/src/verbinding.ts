import type { Status } from "./types";

/** WebSocket naar de backend die zichzelf opnieuw verbindt (500 ms, oplopend tot 5 s). */
export function verbind(opBericht: (s: Status) => void, opVerbinding: (verbonden: boolean) => void): void {
  let wacht = 500;
  const open = () => {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onopen = () => {
      wacht = 500;
      opVerbinding(true);
    };
    ws.onmessage = (e) => opBericht(JSON.parse(e.data) as Status);
    ws.onclose = () => {
      opVerbinding(false);
      setTimeout(open, wacht);
      wacht = Math.min(wacht * 2, 5000);
    };
  };
  open();
}
