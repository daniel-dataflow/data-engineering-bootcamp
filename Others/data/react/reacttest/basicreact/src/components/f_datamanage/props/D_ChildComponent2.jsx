import React from "react";

export default function D_ChildComponent({ children }) {
  //전달된 children데이터 각 변수에 저장해서 관리하기
  //단일값으로 전달된 값을 배열방식이 아니라 구조분해 할당이 불가능함.
  //React.Children.toArray()함수를 이용해서 변환해줘야함.
  const [val, val1, val2, ...other] = Array.isArray(children)
    ? children
    : React.Children.toArray(children);
  return (
    <div>
      <h3>부모가 보낸 데이터</h3>
      <div>{children}</div>
      <h3>구조분해할당으로 받아서 처리하기</h3>
      <div>
        val : {val}
        val1 : {val1}
        val2 : {val2}
      </div>
      <h4>배열함수를 이용해서 출력하기</h4>
      {Array.isArray(children) &&
        children.map((element, i) => <div key={`${i}`}>{element}</div>)}
    </div>
  );
}
