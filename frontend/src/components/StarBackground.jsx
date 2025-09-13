import { useEffect, useRef } from "react";

const StarBackground = () => {
  const canvasRef = useRef(null);
  const stars = useRef([]);
  const mouse = useRef({ x: null, y: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animationFrameId;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Create stars
    const initStars = () => {
      stars.current = Array.from({ length: 100 }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        radius: Math.random() * 2 + 0.5,
        dx: (Math.random() - 0.5) * 0.5,
        dy: (Math.random() - 0.5) * 0.5,
        originalDx: (Math.random() - 0.5) * 0.5,
        originalDy: (Math.random() - 0.5) * 0.5,
      }));
    };
    initStars();

    // Track mouse with proper canvas coordinates
    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.current.x = e.clientX - rect.left;
      mouse.current.y = e.clientY - rect.top;
    };

    window.addEventListener("mousemove", handleMouseMove);

    // Animate
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      stars.current.forEach((star) => {
        // Draw star with glow effect
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        
        // Create gradient for glow effect
        const gradient = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, star.radius * 3);
        gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
        gradient.addColorStop(0.5, "rgba(255, 255, 255, 0.6)");
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
        
        ctx.fillStyle = gradient;
        ctx.fill();

        // Enhanced cursor interaction
        if (mouse.current.x !== null && mouse.current.y !== null) {
          const dx = mouse.current.x - star.x;
          const dy = mouse.current.y - star.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          // Stronger attraction when closer to cursor
          if (dist < 150) {
            const force = Math.max(0, (150 - dist) / 150); // 0 to 1 based on distance
            const attractionStrength = force * 0.15; // Much stronger than before
            
            // Move star towards cursor
            star.x += dx * attractionStrength;
            star.y += dy * attractionStrength;
            
            // Temporarily override natural movement
            star.dx = dx * attractionStrength * 0.3;
            star.dy = dy * attractionStrength * 0.3;
          } else {
            // Gradually return to original movement
            star.dx += (star.originalDx - star.dx) * 0.02;
            star.dy += (star.originalDy - star.dy) * 0.02;
          }
        } else {
          // Return to original movement when no mouse
          star.dx += (star.originalDx - star.dx) * 0.02;
          star.dy += (star.originalDy - star.dy) * 0.02;
        }

        // Natural movement
        star.x += star.dx;
        star.y += star.dy;

        // Wrap around edges
        if (star.x < 0) star.x = canvas.width;
        if (star.x > canvas.width) star.x = 0;
        if (star.y < 0) star.y = canvas.height;
        if (star.y > canvas.height) star.y = 0;
      });

      animationFrameId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="star-background"
    />
  );
};

export default StarBackground;
