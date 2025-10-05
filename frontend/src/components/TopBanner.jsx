import { Link, useLocation } from "react-router-dom";
import foreseelogo from "../assets/foreseelogo.png";

function TopBanner() {
  const location = useLocation();

  return (
    <header className="w-full flex items-center justify-between px-20 py-6 bg-[#EEEDE9] fixed top-0 left-0 z-50 shadow-md">
      {/* Logo */}
      <div className="flex items-center">
        <Link to="/">
          <img
            src={foreseelogo}
            alt="FOREsee Logo"
            className="h-20 object-contain"
          />
        </Link>
      </div>

      {/* Navigation */}
      <nav
        className="flex space-x-10 text-lg font-semibold"
        style={{ color: "#3D57A9" }}
      >
        <Link
          to="/"
          className={
            location.pathname === "/"
              ? "underline underline-offset-4"
              : "hover:underline hover:underline-offset-4"
          }
        >
          Home
        </Link>

        <Link
          to="/foresee"
          className={
            location.pathname === "/foresee"
              ? "underline underline-offset-4"
              : "hover:underline hover:underline-offset-4"
          }
        >
          Foresee
        </Link>

        <Link
          to="/help"
          className={
            location.pathname === "/help"
              ? "underline underline-offset-4"
              : "hover:underline hover:underline-offset-4"
          }
        >
          Need Help?
        </Link>

        <Link
          to="/about"
          className={
            location.pathname === "/about"
              ? "underline underline-offset-4"
              : "hover:underline hover:underline-offset-4"
          }
        >
          About Us
        </Link>
      </nav>
    </header>
  );
}

export default TopBanner;
