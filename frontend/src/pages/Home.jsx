import React, { useState, useEffect } from "react";
import homeimage1 from "../assets/homeimage1.png";

function Home() {
  const steps = [
    {
      title: "1. Upload",
      desc: "Upload the CSV file you want to analyze",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
        </svg>
      ),
    },
    {
      title: "2. Choose",
      desc: "Choose what you want to predict",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125Z" />
        </svg>
      ),
    },
    {
      title: "3. Relax",
      desc: "Sit back while we work on your analysis",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
        </svg>
      ),
    },
    {
      title: "4. Analyze",
      desc: "Your analysis is ready!",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125-.504-1.125 1.125vs16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Zm3.75 11.625a2.625 2.625 0 1 1-5.25 0 2.625 2.625 0 0 1 5.25 0Z" />
        </svg>
      ),
    },
  ];

  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % steps.length);
    }, 2000);
    return () => clearInterval(interval);
  }, [steps.length]);

  const features = [
    {
      key: "smart",
      title: "Smart Predictions",
      short: "Build accurate models effortlessly",
      long:
        "Automatically generate accurate machine learning models without needing deep technical expertise. The system handles preprocessing, training, and optimization so you can focus on results.",
    },
    {
      key: "xai",
      title: "Explainable AI",
      short: "See the “why” behind every result",
      long:
        "Every prediction comes with clear explanations and feature insights. Understand the reasoning behind outcomes to make confident, transparent decisions.",
    },
    {
      key: "integration",
      title: "Easy Integration",
      short: "Connect with your tools in minutes",
      long:
        "Seamlessly connect with existing workflows, tools, and file formats. No complex setup — just upload, integrate, and start analyzing quickly.",
    },
    {
      key: "secure",
      title: "Secure",
      short: "Your data stays safe and private",
      long:
        "Your data is encrypted and never shared. All processing respects privacy standards, ensuring full protection and confidentiality at every step.",
    },
  ];

  const [openFeature, setOpenFeature] = useState(null);

  const toggleFeature = (key) => {
    setOpenFeature((prev) => (prev === key ? null : key));
  };

  return (
    <div className="font-poppin pt-32">

      {/* ✅ Hero Section */}
      <section className="w-full bg-[#EEEDE9] fade-up fade-delay-1">
        <div className="flex flex-col md:flex-row justify-between items-center max-w-6xl mx-auto px-8 py-20 gap-12">
          <div className="max-w-xl">
            <h2 className="text-2xl md:text-3xl font-bold mb-6 text-black">
              Smart analysis, simplified
            </h2>
            <p className="text-gray-700 leading-relaxed">
              Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do
              eiusmod tempor incididunt ut labore et dolore magna aliqua.
            </p>
          </div>

          <div className="mt-10 md:mt-0 w-full md:w-2/5">
            <img
              src={homeimage1}
              alt="Data Visual"
              className="w-full rounded-md shadow-md"
            />
          </div>
        </div>
      </section>

      {/* ✅ How it works Section */}
      <section className="w-full bg-[#D9D7D3] py-16 text-center fade-up fade-delay-2">
        <h2 className="text-3xl font-bold mb-4 text-black">
          Data shouldn’t be too hard to understand!
        </h2>
        <h3 className="text-xl font-semibold mb-12">How it works</h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 px-12 max-w-6xl mx-auto">
          {steps.map((step, index) => (
            <div
              key={step.title}
              className={`flex flex-col items-center text-center space-y-4 shadow-md rounded-xl p-6 transition-all duration-500 ${
                index === activeStep
                  ? "scale-110 bg-[#E6E4DF]"
                  : "scale-100 bg-[#D9D7D3] opacity-80"
              }`}
            >
              {step.icon}
              <h4 className="text-lg font-bold">{step.title}</h4>
              <p className="text-sm text-gray-800">{step.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ✅ Features Section (Accordion) */}
      <section className="w-full bg-[#EEEDE9] py-16 fade-up fade-duration-3">
        <h2 className="text-2xl font-bold mb-8 text-center text-black">Features</h2>

        <div className="max-w-4xl mx-auto px-6 space-y-4">
          {features.map((feature) => {
            const isOpen = openFeature === feature.key;
            return (
              <div key={feature.key} className="rounded-lg bg-[#EEEDE9] border border-[#DDDCDC]">
                <button
                  onClick={() => toggleFeature(feature.key)}
                  className="w-full flex justify-between items-center py-4 px-6 focus:outline-none"
                >
                  <span className="font-semibold text-gray-900">{feature.title}</span>
                  <span className="text-gray-600 text-sm">{feature.short}</span>
                  <svg
                    className={`w-5 h-5 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2}
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                  </svg>
                </button>

                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    isOpen ? "max-h-40 opacity-100" : "max-h-0 opacity-0"
                  }`}
                >
                  <div className="px-6 pb-4 text-gray-700">
                    {feature.long}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-10 text-center">
          <a
            href="/foresee"
            className="bg-[#415AEE] hover:bg-[#3348c8] text-white font-medium py-2 px-6 rounded-lg"
          >
            Try Now
          </a>
        </div>
      </section>
    </div>
  );
}

export default Home;
