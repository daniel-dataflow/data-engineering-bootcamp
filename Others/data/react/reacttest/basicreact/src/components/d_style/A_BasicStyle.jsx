import React from "react";
import "../../assets/basicstyle.css";
export default function A_BasicStyle() {
  const basic = {
    backgroundColor: "magenta",
    color: "white",
    fontSize: "30px",
    width: "300px",
  };
  return (
    <div>
      <h2>인라인로 적용하기</h2>
      <p>
        jsx에 style속성에 객체로 설정한 스타일을 대입, 변수로 스타일을 설정하고
        적용
      </p>
      <p style={basic}>스타일적용하기</p>

      <h2>css파일을 불러와서 처리하기</h2>
      <p>외부 css파일은 assets폴더에 저장하고 import로 불러와서 사용</p>
      <div className="basic">외부파일로 css적용하기</div>
    </div>
  );
}
