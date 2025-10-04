import { useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import TopBanner from "./components/TopBanner";
import Home from "./pages/Home";
import Foresee from "./pages/Foresee";
import About from "./pages/AboutUs";
import AOS from "aos";
import "aos/dist/aos.css";

function App() {
  useEffect(() => {
    AOS.init({
      duration: 900,
      easing: "ease-out",
      once: true,
    });
  }, []);

  return (
    <div className="min-h-screen w-full bg-[#EEEDE9]">
      <TopBanner />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/foresee" element={<Foresee />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </div>
  );
}

export default App;
