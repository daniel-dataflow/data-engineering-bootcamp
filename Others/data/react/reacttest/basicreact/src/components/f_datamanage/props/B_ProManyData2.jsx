import React from "react";
//구조분해할당으로 데이터 처리하기
export default function B_ProManyData2({
  strData,
  numData,
  arrData,
  objData,
  isShow,
  isHidden,
}) {
  return (
    <div>
      <h3>구조분해할당으로 props값 받아서 처리하기</h3>
      <p>가져온 데이터 출력</p>
      <p>strData : {strData}</p>
      <p>numData : {numData}</p>
      <p>arrData : {arrData}</p>
      <p>objData : {Object.keys(objData)}</p>
      <p>isShow : {isShow}</p>
      <p>isHidden : {isHidden}</p>
    </div>
  );
}
