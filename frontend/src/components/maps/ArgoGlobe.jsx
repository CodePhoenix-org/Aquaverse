import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { ArrowLeft, Globe2, Waves } from "lucide-react";
import PageShell from "../ui/PageShell";
import BrandMark from "../ui/BrandMark";

export default function ArgoGlobe() {
  const navigate = useNavigate();
  const mountRef = useRef(null);
  const tooltipRef = useRef(null);
  const mountedRef = useRef(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (mountedRef.current) return undefined;
    mountedRef.current = true;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.1;
    controls.maxDistance = 6;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.18;
    controls.enablePan = false;

    scene.add(new THREE.AmbientLight(0x88aaff, 1.3));
    const sunLight = new THREE.DirectionalLight(0xffffff, 1.8);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    const textureLoader = new THREE.TextureLoader();
    const earthMaterial = new THREE.MeshLambertMaterial({
      map: textureLoader.load(
        "https://raw.githubusercontent.com/turban/webgl-earth/master/images/2_no_clouds_4k.jpg"
      ),
      emissive: new THREE.Color(0x1d4ed8),
      emissiveIntensity: 0.15,
    });

    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(1, 128, 128),
      earthMaterial
    );
    scene.add(earth);

    const dataPointsGroup = new THREE.Group();
    scene.add(dataPointsGroup);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    camera.position.z = 2.5;

    let mouseMoveCleanup = null;

    fetch("/api/argo")
      .then((response) => response.json())
      .then((argoData) => {
        setLoading(false);

        let dataset = argoData;
        if (dataset.length > 5000) {
          dataset = dataset.sort(() => 0.5 - Math.random()).slice(0, 5000);
        }

        const positions = [];
        const colors = [];
        const sizes = [];
        const color = new THREE.Color();
        const floatMeta = [];

        dataset.forEach((point) => {
          const lat = point.latitude;
          const lon = point.longitude;
          const phi = (90 - lat) * (Math.PI / 180);
          const theta = (lon + 180) * (Math.PI / 180);

          const x = -(Math.sin(phi) * Math.cos(theta));
          const y = Math.cos(phi);
          const z = Math.sin(phi) * Math.sin(theta);

          positions.push(x, y, z);
          const hue = 180 + Math.random() * 40;
          color.setHSL(hue / 360, 0.9, 0.58);
          colors.push(color.r, color.g, color.b);
          sizes.push(Math.random() * 6 + 4);

          floatMeta.push({
            ...point,
            pos: new THREE.Vector3(x, y, z),
          });
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3)
        );
        geometry.setAttribute(
          "customColor",
          new THREE.Float32BufferAttribute(colors, 3)
        );
        geometry.setAttribute("size", new THREE.Float32BufferAttribute(sizes, 1));
        geometry.userData.floatMeta = floatMeta;

        const material = new THREE.ShaderMaterial({
          uniforms: { time: { value: 0 } },
          vertexShader: `
            attribute float size;
            attribute vec3 customColor;
            varying vec3 vColor;
            uniform float time;
            void main() {
              vColor = customColor;
              vec3 pos = position;
              float scale = sin(time * 2.0 + pos.x * 10.0) * 0.5 + 1.0;
              gl_PointSize = size * scale;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
            }
          `,
          fragmentShader: `
            varying vec3 vColor;
            void main() {
              float dist = length(gl_PointCoord - vec2(0.5));
              if (dist > 0.5) discard;
              gl_FragColor = vec4(vColor, 1.0);
            }
          `,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthTest: false,
        });

        const points = new THREE.Points(geometry, material);
        dataPointsGroup.add(points);

        const onMouseMove = (event) => {
          mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
          mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);

          const intersects = raycaster.intersectObject(points);
          if (intersects.length > 0) {
            const index = intersects[0].index;
            const meta = geometry.userData.floatMeta[index];
            if (meta && tooltipRef.current) {
              tooltipRef.current.style.display = "block";
              tooltipRef.current.style.left = `${event.clientX + 15}px`;
              tooltipRef.current.style.top = `${event.clientY + 15}px`;
              tooltipRef.current.innerHTML = `
                <strong>${meta.profiler}</strong><br/>
                Lat: ${meta.latitude.toFixed(2)}, Lon: ${meta.longitude.toFixed(2)}<br/>
                ${meta.date_str}
              `;
            }
          } else if (tooltipRef.current) {
            tooltipRef.current.style.display = "none";
          }
        };

        window.addEventListener("mousemove", onMouseMove);
        mouseMoveCleanup = () => window.removeEventListener("mousemove", onMouseMove);
      })
      .catch(() => setLoading(false));

    const animate = (time) => {
      requestAnimationFrame(animate);
      controls.update();
      earth.rotation.y += 0.0003;

      dataPointsGroup.children.forEach((child) => {
        if (child.material?.uniforms) {
          child.material.uniforms.time.value = time * 0.0015;
        }
      });

      renderer.render(scene, camera);
    };
    animate(0);

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      mouseMoveCleanup?.();
      if (mountRef.current?.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <PageShell className="h-screen" contentClassName="h-screen">
      <div className="relative h-screen w-full overflow-hidden">
        <div ref={mountRef} className="absolute inset-0" />

        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(56,189,248,0.06),transparent_38%),linear-gradient(180deg,rgba(2,10,23,0.28),rgba(2,10,23,0.72))]" />

        <header className="absolute left-4 right-4 top-4 z-20 sm:left-6 sm:right-6 lg:left-8 lg:right-8">
          <div className="premium-panel premium-panel-strong flex items-center justify-between gap-4 px-4 py-3 sm:px-6">
            <BrandMark compact />
            <button onClick={() => navigate(-1)} className="premium-button-secondary">
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
          </div>
        </header>

        <div className="absolute left-4 top-24 z-20 max-w-md sm:left-6 sm:top-28 lg:left-8">
          <div className="premium-panel premium-panel-strong p-5 sm:p-6">
            <span className="premium-badge">
              <Globe2 className="h-3.5 w-3.5" />
              3D Global View
            </span>
            <h1 className="mt-5 font-display text-3xl font-bold tracking-[-0.04em] text-white">
              Global BGC Argo floats, reimagined.
            </h1>
            <p className="mt-4 text-sm leading-7 text-slate-300">
              Rotate, zoom, and inspect active float positions through a calmer
              premium interface layered on top of the immersive 3D globe.
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <span className="premium-chip">
                <Waves className="h-4 w-4 text-cyan-100" />
                Live spatial context
              </span>
              <span className="premium-chip">Orbit enabled</span>
            </div>
          </div>
        </div>

        {loading ? (
          <div className="absolute inset-x-0 top-1/2 z-20 mx-auto w-fit -translate-y-1/2 rounded-full border border-white/10 bg-slate-950/70 px-5 py-3 text-sm text-cyan-100 backdrop-blur-xl">
            Loading Argo data...
          </div>
        ) : null}

        <div
          ref={tooltipRef}
          style={{ display: "none" }}
          className="pointer-events-none absolute z-30 rounded-2xl border border-cyan-300/20 bg-slate-950/90 px-4 py-3 text-sm text-slate-100 shadow-ocean backdrop-blur-xl"
        />
      </div>
    </PageShell>
  );
}
