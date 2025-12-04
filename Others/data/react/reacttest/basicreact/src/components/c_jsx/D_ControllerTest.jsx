import React from "react";

export default function C_ConditionTest() {
  const age = 15;
  const msg = "리욕트 조건문";
  //const choice = 2;

  //함수를 선언해서 이용하는 방법
  const conditionTestString = (age) => {
    let msg;
    if (age > 19) {
      msg = "당신은 성인입니다.";
    } else {
      msg = "당신은 성인이 아닙니다.";
    }
    return msg;
  };
  const conditionTestElement = (age) => {
    let msg;
    if (age > 19) {
      msg = "당신은 성인입니다.";
    } else {
      msg = "당신은 성인이 아닙니다.";
    }
    return <h3>{msg}</h3>;
  };
  const switchTest = (cho) => {
    switch (cho) {
      case 1:
        return "1번을 선택하였습니다.";
      case 2:
        return "2번을 선택하였습니다.";
    }
  };
  //반복문 이용하기
  const manyTag = (su) => {
    const tags = [];
    for (let i = 0; i < su; i++) {
      if (i % 2 == 0) tags.push(<p key={i}>{i}</p>);
    }
    return tags;
  };
  return (
    <>
      <h2>조건문사용하기</h2>
      <p>리액트 jsx의 {}내부에서 조건절 if, switch문은 사용할 수 없음</p>
      {/* {
                    if(age>19) <h4>당신은 성인입니다!</h4>
                    else <h4>당신은 미성년입니다.</h4>
                } 사용이불가능함. 컴파일에러 발생함*/}
      {/* {
                    switch(choice){
                        case 1 : <h4>1선택함</h4>;break;
                        case 2 : <h4>2선택함</h4>;break;
                    }
                } switch문도 사용이 불가능함.*/}
      <p>
        조건문을 사용하려면 삼항연산자, 또는 간편연산자(&&, ||)를 이용해서 처리
      </p>
      <h4>삼항연산자 활용</h4>
      {age > 19 ? <h4>당신은 성인입니다.</h4> : <h4>당신은 미성년입니다</h4>}

      <p>단순if조건을 사용할때는 &&, ||를 사용한다.</p>
      <h4>&&연산자 이용하기</h4>
      <p>&& : 조건문의 결과가 true일때 실행하는 구문을 출력하는 로직</p>
      {msg.includes("리액트") && <h4>리액트를 포함한 구문입니다.</h4>}
      <h4>||연산자 이용하기</h4>
      <p>|| : 조건문의 결과가 false일때 실행하는 구문을 출력하는 로직</p>
      {msg.includes("리액트") || <h4>리액트를 포함하지 않은 구문입니다.</h4>}

      <h2>함수를 선언해서 조건문 활용하기</h2>
      <h4>if, switch를 이용해서 처리하는 로직사용</h4>
      <p>함수를 이용해서 처리할 수 있음. 함수의 반환값은 jsx태그도 가능하다.</p>
      {<h4>{conditionTestString(age)}</h4>}
      {conditionTestElement(age)}
      <p>switch이용하기</p>
      <h4>{switchTest(1)}</h4>
      <h4>{switchTest(2)}</h4>
      <h4>다수의 태그를 반환하는 함수 이용하기</h4>
      <p>함수내부에 반복문을 이용해서 처리할 수 있음</p>
      {manyTag(10)}
    </>
  );
}
