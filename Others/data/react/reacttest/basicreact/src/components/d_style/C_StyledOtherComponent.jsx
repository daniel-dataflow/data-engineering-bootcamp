import React from "react";

//styled로 전달된 스타일 적용하기
export default function C_StyledOtherComponent(props) {
  return (
    <div className={props.className}>
      <h2>styled로 전달된 스타일 이용</h2>
      <p>className : {props.className}</p>
      styled적용하기
    </div>
  );
}
