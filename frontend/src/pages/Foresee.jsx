import React from "react";

function Foresee() {
  return (
    <div className="min-h-screen pt-32" style={{ backgroundColor: "#EEEDE9" }}>
      {/* Main Section */}
      <section className="text-center py-11 px-4">
        <h1
          className="text-3xl md:text-3xl font-bold mb-4"
          style={{ color: "#000000" }}
        >
          Analysis Made Easy
        </h1>

        <p className="text-lg text-gray-700 mb-12 max-w-2xl mx-auto">
          Drag and drop a CSV file or select one
        </p>

        {/* Drag & Drop Box */}
        <div
          className="max-w-2xl mx-auto border-2 border-dashed rounded-xl p-16 flex flex-col justify-center items-center"
          style={{
            borderColor: "#3D57A9",
            backgroundColor: "#EEEDE9",
          }}
        >
          <div className="flex justify-center mb-6">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="w-24 h-24 text-[#5A6ACF]"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
              />
            </svg>
          </div>
        </div>

        {/* Submit Button */}
        <div className="mt-6">
          <button
            className="px-12 py-2 rounded-md text-white"
            style={{ backgroundColor: "#3D57A9" }}
          >
            Submit
          </button>
        </div>
      </section>
    </div>
  );
}

export default Foresee;
