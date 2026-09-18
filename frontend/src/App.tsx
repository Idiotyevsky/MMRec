import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import LiveDemo from "./pages/LiveDemo";
import Inspector from "./pages/Inspector";
import ColdStart from "./pages/ColdStart";
import System from "./pages/System";

const NAV = [
  { to: "/", label: "Live Demo" },
  { to: "/inspect", label: "Inspector" },
  { to: "/cold", label: "Cold Start" },
  { to: "/system", label: "System" },
];

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          ShortRec<span>multimodal two-stage recommendation</span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => (isActive ? "active" : "")}>
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="content">
        <Routes>
          <Route path="/" element={<LiveDemo />} />
          <Route path="/inspect" element={<Inspector />} />
          <Route path="/cold" element={<ColdStart />} />
          <Route path="/system" element={<System />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
