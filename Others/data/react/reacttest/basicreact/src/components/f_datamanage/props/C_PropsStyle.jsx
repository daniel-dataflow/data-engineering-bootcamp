import React from "react";

export default function C_StyleProp({ style }) {
  return (
    <div>
      <h3>style데이터를 외부에서 받아서 처리하기</h3>
      <p>style을 적용하는 객체를 전달받아서 구현하기</p>
      <p style={style}>다양한 스타일 적용하기</p>
    </div>
  );
}
