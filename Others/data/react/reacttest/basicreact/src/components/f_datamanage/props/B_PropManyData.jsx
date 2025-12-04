import React from "react";
//props객체로 전달받아서 처리하기
export default function B_PropManyData(props) {
  return (
    <div>
      <h3>다양한 데이터를 전달받아 처리하기</h3>
      <p>
        model에 저장된 데이터를 전달받아 처리하기 model에 저장된 값은 {"{}"}를
        이용해서 변수명을 지정
      </p>
      <h4>일반값을 출력하기</h4>
      <p>strData : {props.strData}</p>
      <p>numData : {props.numData}</p>
      <p>
        isShow : {props.isShow} {typeof props.isShow}
      </p>
      <p>
        boolean형은 화면에 값이 출력되지 않고 특정태그를 출력할지를 결정하는
        조건문으로 활용할 수 있음 삼항연산자나 간편연산자를 활용할 수 있음
      </p>
      <p>true일때 : {props.isShow && props.numData}</p>
      <p>fals일때 : {props.isHidden && props.strData}</p>
      <p>삼항연산자 : {props.isShow ? "보여줘" : "보여주지마"}</p>

      <h4>객체, 배열 출력하기</h4>
      <p>
        객체, 배열형태로 props로 전달된 값은 직접 접근하거나 함수(map, keys,
        values 등)를 이용해서 나열하여 출력함. -> 함수는 반환형이 있어야 출력이
        가능.
      </p>
      <p>arrData[0] : {props.arrData[0]}</p>
      <p>arrData[1] : {props.arrData[1]}</p>
      <p>arrData[2] : {props.arrData[2]}</p>
      <p>objData.name : {props.objData.name}</p>
      <p>objData.age : {props.objData.age}</p>
      <p>objData.address : {props.objData.address}</p>
      <h4>함수활용해서 데이터 출력하기</h4>
      <p>배열값 출력</p>
      <ul>
        {props.arrData.map((num, i) => (
          <li key={i}>{num}</li>
        ))}
      </ul>
      <p>객체값 출력</p>
      <p>객체 key값을 출력하기 -> Object.keys함수를 이용</p>
      {Object.keys(props.objData).map((key, i) => (
        <p key={`${key}_${i}`}>{key}</p>
      ))}
      <p>객체 value값을 출력하기 -> Object.values함수를 이용</p>
      {Object.values(props.objData).map((value, i) => (
        <p key={`${value}_${i}`}>{value}</p>
      ))}

      <h4>props객체의 데이터를 탐색하기</h4>
      <p>key값을 가져오기</p>
      <ul>
        {Object.keys(props).map((key, i) => (
          <li key={`${key}_${i}`}>{key}</li>
        ))}
      </ul>
      <p>value값을 가져오기</p>
      <ul>
        {Object.values(props).map((val, i) => {
          return (typeof val != "function" && typeof val != "object") ||
            Array.isArray(val) ? (
            <li key={`${val}_${i}`}>
              {val} {typeof val}
            </li>
          ) : (
            <li key={`${val}_${i}`}>출력못함 {typeof val}</li>
          );
        })}
      </ul>
    </div>
  );
}
