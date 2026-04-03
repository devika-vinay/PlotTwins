import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import BusinessEventPlanner from "./pages/BusinessEventPlanner";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/business" element={<BusinessEventPlanner />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);