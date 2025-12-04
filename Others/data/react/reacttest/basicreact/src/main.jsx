import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  // 잠재적인 문제를 식별하고 경고를 해주는 개발 도구 -> React개발관행에 따라 개발할 수 있도록 하기 위해 설정, 개발시에만 발생 production에서는 작동하지 않음
  // 미래코드대비, 문제점 조기발견, 안전하지 않은 생명주기 메소드식별
  <StrictMode>
    <App />
  </StrictMode>
);
