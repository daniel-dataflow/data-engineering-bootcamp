import React from "react";

// export default function A_PropsFunction(props) {
export default function A_PropsFunction({ title }) {
  //prop데이터를 구조분해할당으로 데이터를 받으면
  //지역변수로 수정은 가능하나 수정값이 화면에 반영되지 않음(값이 변경된다고 화면이 리랜더링 되지 않음) -> 반응성이 없는 데이터
  const changeProps = (e) => {
    // props.title += "변경하기"; //Uncaught TypeError: Cannot assign to read only property 'title' of object '#<Object>' 에러 발생
    title += "변경하기";
    console.log(title);
  };
  return (
    <div>
      <h3>함수형컴포넌트에서 활용하기</h3>
      <p>함수형 컴포넌트에 매개변수로 props값을 받아서 활용할 수 있음</p>
      <p>
        {/* props -> {Object.keys(props)} : {Object.values(props)} */}
        props -> {title}
      </p>
      <button onClick={changeProps}>prop변경하기</button>
    </div>
  );
}
