import React from "react";
import "../../assets/header.css";
import header from "../../assets/header.module.css";

export default function B_Header() {
  return (
    <div>
      <h3>header.css에서 적용한 title</h3>
      <p className="title">header제목</p>

      <h3>header 모듈로 적용한 스타일</h3>
      <div className={header.container}>header 스타일적용</div>
    </div>
  );
}
