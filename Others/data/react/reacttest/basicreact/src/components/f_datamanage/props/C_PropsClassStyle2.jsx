import React from "react";

export default function C_ClassStyleProps({ classStyle }) {
  return (
    <div>
      <h4>클래스를 전달하여 스타일 적용</h4>
      <p>tailwindcss 클래스를 전달해서 적용할 수 있음</p>
      <p className={classStyle.join(" ")}>class로 스타일적용</p>
    </div>
  );
}
