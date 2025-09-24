import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

const ArgoGlobe = () => {
  const mountRef = useRef();
  const tooltipRef = useRef();
  const mountedRef = useRef(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
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

    // Lights (bright, no specular highlights)
    scene.add(new THREE.AmbientLight(0x8888ff, 1.2));
    const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
    sunLight.position.set(5, 3, 5);
    scene.add(sunLight);

    // Earth (with continents and oceans)
    const textureLoader = new THREE.TextureLoader();
    const earthMaterial = new THREE.MeshLambertMaterial({
      map: textureLoader.load(
        "https://raw.githubusercontent.com/turban/webgl-earth/master/images/2_no_clouds_4k.jpg"
      ),
      emissive: new THREE.Color(0x222222), // Slight emissive for brightness
      emissiveIntensity: 0.3,
    });
    const earth = new THREE.Mesh(
      new THREE.SphereGeometry(1, 128, 128),
      earthMaterial
    );
    scene.add(earth);

    // Data Points group
    const dataPointsGroup = new THREE.Group();
    scene.add(dataPointsGroup);
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    camera.position.z = 2.5;

    // FETCH FLOAT DATA
    fetch("/api/argo")
      .then((res) => res.json())
      .then((argoData) => {
        setLoading(false);

        if (argoData.length > 5000) {
          argoData = argoData.sort(() => 0.5 - Math.random()).slice(0, 5000);
        }

        const positions = [];
        const colors = [];
        const sizes = [];
        const color = new THREE.Color();
        const floatMeta = []; // store metadata

        argoData.forEach((point) => {
          const lat = point.latitude;
          const lon = point.longitude;
          const phi = (90 - lat) * (Math.PI / 180);
          const theta = (lon + 180) * (Math.PI / 180);

          const x = -(Math.sin(phi) * Math.cos(theta));
          const y = Math.cos(phi);
          const z = Math.sin(phi) * Math.sin(theta);

          positions.push(x, y, z);
          const hue = Math.random() * 50; // yellow-red neon
          color.setHSL(hue / 360, 1, 0.5);
          colors.push(color.r, color.g, color.b);
          sizes.push(Math.random() * 6 + 4);

          // save metadata
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
        geometry.setAttribute(
          "size",
          new THREE.Float32BufferAttribute(sizes, 1)
        );

        // attach metadata to geometry
        geometry.userData.floatMeta = floatMeta;

        const material = new THREE.ShaderMaterial({
          uniforms: {
            time: { value: 0 },
          },
          vertexShader: `
            attribute float size;
            attribute vec3 customColor;
            varying vec3 vColor;
            uniform float time;
            void main() {
              vColor = customColor;
              vec3 pos = position;
              float scale = sin(time*2.0 + pos.x*10.0) * 0.5 + 1.0;
              gl_PointSize = size * scale;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(pos,1.0);
            }
          `,
          fragmentShader: `
            varying vec3 vColor;
            void main() {
              float dist = length(gl_PointCoord - vec2(0.5));
              if(dist > 0.5) discard;
              gl_FragColor = vec4(vColor, 1.0);
            }
          `,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthTest: false,
        });

        const points = new THREE.Points(geometry, material);
        dataPointsGroup.add(points);

        // mouse hover tooltip
        const onMouseMove = (event) => {
          mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
          mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

          raycaster.setFromCamera(mouse, camera);

          const intersects = raycaster.intersectObject(points);
          if (intersects.length > 0) {
            const index = intersects[0].index;
            const meta = geometry.userData.floatMeta[index];
            if (meta) {
              tooltipRef.current.style.display = "block";
              tooltipRef.current.style.left = event.clientX + 15 + "px";
              tooltipRef.current.style.top = event.clientY + 15 + "px";
              tooltipRef.current.innerHTML = `
                <b>${meta.profiler}</b><br/>
                📍 Lat: ${meta.latitude.toFixed(2)}, Lon: ${meta.longitude.toFixed(2)}<br/>
                🗓️ ${meta.date_str}
              `;
            }
          } else {
            tooltipRef.current.style.display = "none";
          }
        };
        window.addEventListener("mousemove", onMouseMove);

        // cleanup
        return () => {
          window.removeEventListener("mousemove", onMouseMove);
        };
      })
      .catch((err) => console.error("Failed to fetch Argo data:", err));

    // animate
    const animate = (time) => {
      requestAnimationFrame(animate);
      controls.update();
      earth.rotation.y += 0.0003;

      dataPointsGroup.children.forEach((child) => {
        if (child.material && child.material.uniforms) {
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
      if (mountRef.current) mountRef.current.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <>
      {/* Info panel */}
      <div
        className="mt-10"
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(0,0,0,0.7)",
          color: "#FFD700",
          padding: "12px 16px",
          borderRadius: "8px",
          fontSize: "14px",
          lineHeight: "1.5",
          maxWidth: "280px",
          zIndex: 10,
        }}
      >
        <b>Global BGC Argo Floats</b>
        <br />
        Explore the latest locations of active BGC floats around the world. This
        interactive 3D globe lets you rotate, zoom, and discover oceanographic
        data in real-time, helping you visualize global ocean monitoring
        effortlessly.
      </div>

      {loading && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            color: "cyan",
          }}
        >
          Loading Argo Data...
        </div>
      )}
      <div ref={mountRef} />
      <div
        ref={tooltipRef}
        style={{
          position: "absolute",
          display: "none",
          padding: "10px",
          background: "rgba(26,0,51,0.85)",
          border: "1px solid #ff00ff",
          borderRadius: "5px",
          pointerEvents: "none",
          fontSize: "14px",
          color: "#fff",
        }}
      />
    </>
  );
};

export default ArgoGlobe;