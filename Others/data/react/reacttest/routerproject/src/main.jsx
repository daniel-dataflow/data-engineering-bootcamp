import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { BrowserRouter } from "react-router-dom";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    {/* 라우터를 적용하기 위해 base component를 root에 설정해줘야 함 */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>
);
