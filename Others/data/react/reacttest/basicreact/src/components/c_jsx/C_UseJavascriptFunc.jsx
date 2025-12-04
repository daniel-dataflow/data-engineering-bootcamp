import React from "react";

export default function C_UseJavascriptFunc() {
  //커스텀함수를 이용해서 화면구성하기
  //함수는 리터럴 값이나 jsx객체를 반환하는 함수를 생성
  const basicFunc = () => {
    return "고정값반환";
  };
  const basicFuncSu = () => {
    return 100;
  };
  //빈값으로 출력됨.
  const basicFuncNone = () => {};
  const basicArray = () => {
    return [1, 2, 3, 4, 5];
  };
  const basicJSX = () => {
    const msg = "jsx를 이용하기";
    return <span>{msg}</span>;
  };
  //다수 JSX보내기 list, table
  const makeList = () => {
    const arr = ["유병승", "홍길동", "고길동"];
    //다수는 배열을 만들어서 반환해야함.
    return arr.map((n) => <li key={n}>{n}</li>);
  };

  return (
    <>
      <h3>커스텀함수이용하기</h3>
      <p>함수를 이용할때는 반드시 반환형이 있어야함</p>
      <h4>리터럴을 반환하는 함수 이용하기</h4>
      <p>출력 : {basicFunc()}</p>
      <p>출력 : {basicFuncSu()}</p>
      <p>출력 : {basicFuncNone()}</p>
      <p>출력 : {basicArray()}</p>
      <p>출력 : {basicArray()[0]}</p>
      <h4>jsx를 반환하는 함수 이용하기</h4>
      <p>JSX를 반환하는 함수를 만들어서 이용할 수 있음</p>
      <p>
        JSX를 반환할때 문법은 동일하게 작성되며 js구문을 작성할때는 {}로
        표시해줘야함.
      </p>
      <p>{basicJSX()}</p>
      <ul>{makeList()}</ul>
    </>
  );
}
