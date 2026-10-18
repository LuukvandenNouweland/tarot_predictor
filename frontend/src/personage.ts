import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { ANIMATIES } from "./animaties";

/** three.js-scene met het personage en zijn animaties. */
export class Personage {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  private klok = new THREE.Clock();
  private mixer: THREE.AnimationMixer | null = null;
  private acties = new Map<string, THREE.AnimationAction>();
  private huidige: THREE.AnimationAction | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    // Licht voor geïntegreerde GPU's: pixelratio max 2, geen schaduwen.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2));
    const zon = new THREE.DirectionalLight(0xffffff, 1.5);
    zon.position.set(2, 4, 3);
    this.scene.add(zon);

    new ResizeObserver(() => this.pasGrootteAan()).observe(container);
    this.pasGrootteAan();
    this.renderer.setAnimationLoop(() => {
      this.mixer?.update(this.klok.getDelta());
      this.renderer.render(this.scene, this.camera);
    });
  }

  async laad(url: string): Promise<void> {
    const gltf = await new GLTFLoader().loadAsync(url);
    this.scene.add(gltf.scene);
    this.richtCamera(gltf.scene);

    this.mixer = new THREE.AnimationMixer(gltf.scene);
    for (const clip of gltf.animations) {
      this.acties.set(clip.name, this.mixer.clipAction(clip));
    }
    console.log("Animaties in het model:", gltf.animations.map((c) => c.name));
    // Na een eenmalige animatie terug naar idle.
    this.mixer.addEventListener("finished", (e) => {
      if (e.action === this.huidige) this.speelIdle();
    });
    this.speelIdle();
  }

  speelIdle(): void {
    this.speel(ANIMATIES.idle, false);
  }

  /** Speelt een animatie één keer en gaat daarna terug naar idle. */
  speelEenmalig(naam: string): void {
    this.speel(naam, true);
  }

  private speel(naam: string, eenmalig: boolean): void {
    const actie = this.acties.get(naam);
    if (!actie) {
      console.warn(`Animatie "${naam}" bestaat niet. Beschikbaar:`, [...this.acties.keys()]);
      return;
    }
    if (actie === this.huidige && !eenmalig) return;

    actie.reset();
    actie.setLoop(eenmalig ? THREE.LoopOnce : THREE.LoopRepeat, Infinity);
    actie.clampWhenFinished = eenmalig;
    actie.play();
    if (this.huidige && this.huidige !== actie) {
      this.huidige.crossFadeTo(actie, ANIMATIES.overgang, false);
    }
    this.huidige = actie;
  }

  /** Camera zo zetten dat het hele model in beeld is, wat het formaat ook is. */
  private richtCamera(model: THREE.Object3D): void {
    const doos = new THREE.Box3().setFromObject(model);
    const midden = doos.getCenter(new THREE.Vector3());
    const maat = doos.getSize(new THREE.Vector3());
    const afstand = (Math.max(maat.x, maat.y, maat.z) / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.6;
    this.camera.position.set(midden.x, midden.y + maat.y * 0.15, midden.z + afstand);
    this.camera.near = afstand / 100;
    this.camera.far = afstand * 100;
    this.camera.lookAt(midden);
    this.camera.updateProjectionMatrix();
  }

  private pasGrootteAan(): void {
    const { clientWidth: b, clientHeight: h } = this.container;
    if (b === 0 || h === 0) return;
    this.renderer.setSize(b, h, false);
    this.camera.aspect = b / h;
    this.camera.updateProjectionMatrix();
  }
}
