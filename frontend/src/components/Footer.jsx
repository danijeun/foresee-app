import React from "react";

export default function Footer() {
  return (
    <footer className="text-center py-6 text-sm text-gray-600 fade-up fade-delay-7">
      © {new Date().getFullYear()} Foresee — Made with ❤️ by the team
    </footer>
  );
}
