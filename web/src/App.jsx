import { useEffect, useState } from "react";
import Landing from "./components/Landing.jsx";
import ChatApp from "./components/ChatApp.jsx";

function readLang() {
  try {
    const saved = localStorage.getItem("asov-lang");
    if (saved === "es" || saved === "en") return saved;
  } catch {
    /* storage may be blocked */
  }
  return "es";
}

const routeOf = () => (window.location.hash.startsWith("#/app") ? "app" : "landing");

export default function App() {
  const [lang, setLangState] = useState(readLang);
  const [route, setRoute] = useState(routeOf);

  useEffect(() => {
    const onHash = () => {
      setRoute(routeOf());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = (value) => {
    setLangState(value);
    try {
      localStorage.setItem("asov-lang", value);
    } catch {
      /* ignore */
    }
  };

  const go = (target) => {
    window.location.hash = String(target).includes("app") ? "#/app" : "#/";
  };

  return route === "app" ? (
    <ChatApp lang={lang} setLang={setLang} go={go} />
  ) : (
    <Landing lang={lang} setLang={setLang} go={go} />
  );
}
