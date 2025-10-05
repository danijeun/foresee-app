import React from "react";
import foresightImg from "../assets/foresight.png";
import help1 from "../assets/help1.png";
import homeimage2 from "../assets/homeimage2.png";
import RicardoImg from "../assets/RicardoMejia.jpeg";
import MarianoImg from "../assets/MarianoRamos.jpg";
import DanielImg from "../assets/DanielJeun.jpg";
import EmiletteImg from "../assets/EmiletteSegura.jpg";
import { useNavigate } from "react-router-dom";
import Footer from "../components/Footer";


export default function About() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen font-poppins bg-[#EEEDE9] text-black pt-32">
      
      {/* 1️⃣ Header Section */}
      <section className="text-center pt-8 pb-2 px-6 fade-up fade-delay-1">
        <h1 className="text-4xl font-bold mb-2">About ForeSee</h1>
        <p className="text-lg max-w-2xl mx-auto leading-relaxed mb-1">
          Empowering anyone to understand data.
        </p>
        <p className="text-lg max-w-2xl mx-auto leading-relaxed flex items-center justify-center gap-2">
          Your data, your{" "}
          <img
            src={foresightImg}
            alt="foresight"
            className="inline-block h-7 md:h-8 lg:h-9 object-contain"
          />
        </p>
      </section>

      {/* Animated Plane */}
      <section className="relative w-full h-32 md:h-36 lg:h-40 mb-2 overflow-hidden fade-up fade-delay-1">
        {/* SVG remains unchanged */}
        <svg
          viewBox="0 0 1200 240"
          preserveAspectRatio="none"
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* (same SVG code as before) */}
          <defs>
            <mask id="revealMask">
              <rect width="100%" height="100%" fill="black" />
              <path
                id="motionPath"
                d="M -80 170
                  C 200 90, 500 230, 820 110
                  S 1120 230, 1280 150"
                fill="none"
                stroke="white"
                strokeWidth="14"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeDasharray="2000"
                strokeDashoffset="2000"
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="2000"
                  to="0"
                  dur="4.3s"
                  fill="freeze"
                />
              </path>
            </mask>
          </defs>

          <path
            d="M -80 170
              C 200 90, 500 230, 820 110
              S 1120 230, 1280 150"
            fill="none"
            stroke="#1E3A8A"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="8 12"
            mask="url(#revealMask)"
          />

          <g transform="translate(-10,-10)">
            <path
              d="M2.25 2.25l19.5 9.75-19.5 9.75L2.25 14.25l13.5-2.25-13.5-2.25V2.40z"
              fill="#1E3A8A"
            >
              <animateMotion dur="2.95s" fill="freeze" rotate="auto" begin="0s">
                <mpath href="#motionPath" />
              </animateMotion>
            </path>
          </g>
        </svg>
      </section>

      {/* 2️⃣ Mission Section */}
      <section className="relative overflow-hidden bg-[#E6E4DF] py-16 fade-up fade-delay-2">
        <div className="absolute inset-0 chevron-overlay pointer-events-none">
          <div></div>
          <div></div>
          <div></div>
        </div>

        <div className="flex flex-col md:flex-row items-center justify-center px-10 md:px-20 gap-16 relative z-10">
          <div className="md:w-1/2 text-center md:text-left">
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2 justify-center md:justify-start">
              <span role="img" aria-label="lightbulb">💡</span> Our Mission
            </h2>
            <p className="text-lg leading-relaxed">
              At Foresee, we simplify the power of prediction and analytics.
              We believe foresight shouldn't be limited to analysts — anyone should
              be able to understand and make decisions with clarity and confidence.
            </p>
          </div>

          <div className="md:w-1/2 flex justify-center">
            <img
              src={help1}
              alt="Mission Illustration"
              className="w-[450px] h-auto rounded-xl object-contain"
            />
          </div>
        </div>
      </section>

      {/* 3️⃣ Our Story Section */}
      <section className="relative overflow-hidden bg-[#DDDAD3] py-16 fade-up fade-delay-3">
        <div className="absolute inset-0 chevron-overlay-inverse pointer-events-none">
          <div></div>
          <div></div>
          <div></div>
        </div>

        <div className="flex flex-col md:flex-row-reverse items-center justify-between px-10 md:px-20 gap-10 relative z-10">
          <div className="md:w-1/2">
            <h2 className="text-2xl font-semibold mb-4">Our Story</h2>
            <p className="leading-relaxed">
              Foresee was born from a desire to make complex data insights 
              intuitive. We started with one goal: turn prediction tools 
              into something anyone can use without prior experience.
            </p>
          </div>

          <div className="md:w-1/2 flex justify-center">
            <img
              src={homeimage2}
              alt="Our Story Illustration"
              className="w-[450px] h-auto rounded-xl object-contain"
            />
          </div>
        </div>
      </section>

      {/* 4️⃣ Meet the Team */}
      <section className="relative overflow-hidden bg-[#E6E4DF] py-16 px-10 md:px-20 text-center fade-up fade-delay-4">
        <div className="absolute inset-0 chevron-overlay pointer-events-none">
          <div></div>
          <div></div>
          <div></div>
        </div>

        <div className="relative z-10">
          <h2 className="text-2xl font-semibold mb-10">Meet the Team</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10">
            {[
              { name: "Ricardo Mejia", role: "Computer Science Major", desc: "Back-End Developer", img: RicardoImg },
              { name: "Daniel Jeun", role: "Data Science Major", desc: "Back-End Developer", img: DanielImg },
              { name: "Mariano Ramos", role: "Computer Science Major", desc: "Front-End Developer", img: MarianoImg },
              { name: "Emilette Segura", role: "IT Major", desc: "UX/UI Designer", img: EmiletteImg },
            ].map((member, index) => (
              <div
                key={index}
                className="bg-white rounded-xl shadow-md p-6 flex flex-col items-center"
              >
                <img
                  src={member.img}
                  alt={member.name}
                  className="w-24 h-24 rounded-full object-cover mb-4"
                />
                <h3 className="font-semibold">{member.name}</h3>
                <p className="text-sm text-gray-600">{member.role}</p>
                <p className="text-sm mt-2">{member.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 5️⃣ Our Values */}
      <section className="relative overflow-hidden bg-[#DDDAD3] py-16 px-10 md:px-20 text-center fade-up fade-delay-5">
        <div className="absolute inset-0 chevron-overlay-inverse pointer-events-none">
          <div></div>
          <div></div>
          <div></div>
        </div>

        <div className="relative z-10">
          <h2 className="text-2xl font-semibold mb-10">Our Values</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
            {[
              { title: "Simplicity", desc: "We turn complexity into clarity through minimal design and usability." },
              { title: "Transparency", desc: "We build trust with clear insights and honest communication." },
              { title: "Impact", desc: "We empower users to take action that matters through foresight." }
            ].map((value, index) => (
              <div
                key={index}
                className="bg-white rounded-xl shadow-md p-6 text-center"
              >
                <h3 className="font-semibold text-lg mb-2">{value.title}</h3>
                <p className="text-sm leading-relaxed">{value.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 6️⃣ Call to Action */}
      <section className="text-center py-16 fade-up fade-delay-6">
        <h2 className="text-2xl font-semibold mb-6">Ready to foresee?</h2>
        <button
          onClick={() => navigate("/foresee")}
          className="px-6 py-3 bg-black text-white rounded-xl hover:opacity-80 transition"
        >
          Try Demo
        </button>
      </section>

      <Footer />

    </div>
  );
}
