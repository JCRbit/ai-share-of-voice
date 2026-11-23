// Small motion helpers. Everything here does nothing when the person asked for reduced motion.
import { useEffect, useRef, useState } from "react";

export const reducedMotion = () => {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
};

// Adds the class "in" to every [data-reveal] element inside the ref once it scrolls into view.
export function useReveal(deps = []) {
  const ref = useRef(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return undefined;
    const items = Array.from(root.querySelectorAll("[data-reveal]:not(.in)"));
    if (reducedMotion() || typeof IntersectionObserver === "undefined") {
      items.forEach((el) => el.classList.add("in"));
      return undefined;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      }),
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" }
    );
    items.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return ref;
}

// A number that counts up to its value the first time it becomes visible.
export function CountUp({ value, duration = 1100, decimals = 0 }) {
  const ref = useRef(null);
  const [shown, setShown] = useState(reducedMotion() ? value : 0);
  useEffect(() => {
    if (reducedMotion()) { setShown(value); return undefined; }
    let raf = 0;
    let started = false;
    const run = () => {
      started = true;
      const t0 = performance.now();
      const step = (now) => {
        const p = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        setShown(value * eased);
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") { run(); return () => cancelAnimationFrame(raf); }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting && !started) { run(); io.disconnect(); } }, { threshold: 0.4 });
    io.observe(el);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  }, [value, duration]);
  return <span ref={ref}>{shown.toFixed(decimals)}</span>;
}

// Types a string out, one character at a time. Returns [text so far, finished].
export function useTypewriter(text, speed = 28, startDelay = 0) {
  const [n, setN] = useState(reducedMotion() ? text.length : 0);
  useEffect(() => {
    if (reducedMotion()) { setN(text.length); return undefined; }
    setN(0);
    let i = 0;
    let timer = setTimeout(function tick() {
      i += 1;
      setN(i);
      if (i < text.length) timer = setTimeout(tick, speed + (/[,.¿?]/.test(text[i - 1]) ? speed * 4 : 0));
    }, startDelay);
    return () => clearTimeout(timer);
  }, [text, speed, startDelay]);
  return [text.slice(0, n), n >= text.length];
}
