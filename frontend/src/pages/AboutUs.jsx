import React from "react";

export default function About() {
  return (
    <div className="min-h-screen font-poppins bg-[#EEEDE9] text-black pt-32">
      {/* Header Section */}
      <section className="text-center py-16 px-6">
        <h1 className="text-4xl font-bold mb-4">About Foresee</h1>
        <p className="text-lg max-w-2xl mx-auto leading-relaxed">
          Empowering anyone to understand data. <br />
          Your data, your foresight.
        </p>
      </section>

      {/* Mission Section */}
      <section className="flex flex-col md:flex-row items-center justify-between px-10 md:px-20 py-16 gap-10">
        <div className="md:w-1/2">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <span role="img" aria-label="lightbulb">💡</span> Our Mission
          </h2>
          <p className="leading-relaxed">
            At Foresee, we simplify the power of prediction and analytics.
            We believe foresight shouldn't be limited to analysts — 
            anyone should be able to understand and make decisions 
            with clarity and confidence.
          </p>
        </div>
        <div className="md:w-1/2 h-56 bg-gray-300 rounded-xl flex items-center justify-center">
          <span className="text-gray-600">Image Placeholder</span>
        </div>
      </section>

      {/* History Section */}
      <section className="flex flex-col md:flex-row-reverse items-center justify-between px-10 md:px-20 py-16 gap-10">
        <div className="md:w-1/2">
          <h2 className="text-2xl font-semibold mb-4">Our Story</h2>
          <p className="leading-relaxed">
            Foresee was born from a desire to make complex data insights 
            intuitive. We started with one goal: turn prediction tools 
            into something anyone can use without prior experience.
          </p>
        </div>
        <div className="md:w-1/2 h-56 bg-gray-300 rounded-xl flex items-center justify-center">
          <span className="text-gray-600">Image Placeholder</span>
        </div>
      </section>

      {/* Team Section */}
      <section className="px-10 md:px-20 py-16 text-center">
        <h2 className="text-2xl font-semibold mb-10">Meet the Team</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10">
          {[
            { name: "Member 1", role: "Role", desc: "Short description." },
            { name: "Member 2", role: "Role", desc: "Short description." },
            { name: "Member 3", role: "Role", desc: "Short description." },
            { name: "Member 4", role: "Role", desc: "Short description." },
          ].map((member, index) => (
            <div
              key={index}
              className="bg-white rounded-xl shadow-md p-6 flex flex-col items-center"
            >
              <div className="w-24 h-24 bg-gray-300 rounded-full mb-4" />
              <h3 className="font-semibold">{member.name}</h3>
              <p className="text-sm text-gray-600">{member.role}</p>
              <p className="text-sm mt-2">{member.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Values Section */}
      <section className="px-10 md:px-20 py-16 text-center">
        <h2 className="text-2xl font-semibold mb-10">Our Values</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
          {[
            {
              title: "Simplicity",
              desc: "We turn complexity into clarity through minimal design and usability."
            },
            {
              title: "Transparency",
              desc: "We build trust with clear insights and honest communication."
            },
            {
              title: "Impact",
              desc: "We empower users to take action that matters through foresight."
            }
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
      </section>

      {/* Call to Action */}
      <section className="text-center py-16">
        <h2 className="text-2xl font-semibold mb-6">Ready to foresee?</h2>
        <button className="px-6 py-3 bg-black text-white rounded-xl hover:opacity-80 transition">
          Try Demo
        </button>
      </section>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-gray-600">
        © {new Date().getFullYear()} Foresee — Made with ❤️ by the team
      </footer>
    </div>
  );
}
