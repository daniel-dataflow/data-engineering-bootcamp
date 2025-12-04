import React from "react";
import "../../assets/footer.css";
import footer from "../../assets/footer.module.css";
export default function B_Footer() {
  return (
    <div>
      <h3>footer.css에서 적용한 title</h3>
      <p className="title">footer제목</p>

      <h3>footer 모듈로 적용한 스타일</h3>
      <div className={footer.container}>footer 스타일적용</div>
    </div>
  );
}
