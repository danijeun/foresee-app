import React from "react";
import homeimage1 from "../assets/homeimage1.png";

function Home() {
  return (
    <div className="font-poppin pt-32">
      
      {/* ✅ Hero Section (full width) */}
      <section className="w-full bg-[#EEEDE9]" data-aos="fade-up">
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
      <section className="w-full bg-[#D9D7D3] py-16 text-center" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-4 text-black">
          Data shouldn’t be too hard to understand!
        </h2>
        <h3 className="text-xl font-semibold mb-12">How it works</h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 px-12 max-w-6xl mx-auto">

          {/* Step 1 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "0s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
            <h4 className="text-lg font-bold">1. Upload</h4>
            <p className="text-sm text-gray-800">Upload the CSV file you want to analyze</p>
          </div>

          {/* Step 2 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "2.5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125Z" />
            </svg>
            <h4 className="text-lg font-bold">2. Choose</h4>
            <p className="text-sm text-gray-800">Choose what you want to predict</p>
          </div>

          {/* Step 3 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
            </svg>
            <h4 className="text-lg font-bold">3. Relax</h4>
            <p className="text-sm text-gray-800">Sit back while we work on your analysis</p>
          </div>

          {/* Step 4 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "7.5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Zm3.75 11.625a2.625 2.625 0 1 1-5.25 0 2.625 2.625 0 0 1 5.25 0Z" />
            </svg>
            <h4 className="text-lg font-bold">4. Analyze</h4>
            <p className="text-sm text-gray-800">Your analysis is ready!</p>
          </div>

        </div>
      </section>

    </div>
  );
}

export default Home;
