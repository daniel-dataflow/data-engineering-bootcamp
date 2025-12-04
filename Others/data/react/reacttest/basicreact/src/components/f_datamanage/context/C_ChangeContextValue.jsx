import React, { useContext } from "react";
import { ChangeContext } from "./resources/Context";
export default function C_ChangeContextValue() {
  let changeData = useContext(ChangeContext);
  return (
    <div>
      <h4>ChangeContextValue1</h4>
      <h5>context값 수정하기</h5>
      <p>
        일반데이터는 let으로 받아서 수정해도 리랜더링 되어 변경되지 않음
        <br />
        state와 연동된 데이터는 수정하면 리랜더링되어 수정된 값이 페이지에
        반영됨.
      </p>
      {typeof changeData == "string" ? (
        <>
          <p>changeData : {changeData}</p>
          <input
            type="text"
            onChange={(e) => {
              //변경해도 반응성이 없어 리랜더링이 되지 않음
              changeData = e.target.value;
              console.log(changeData); //console에 출력은 가능함.
            }}
          />
        </>
      ) : (
        <>
          <p>changeData.data : {changeData.data}</p>
          <input
            type="text"
            onChange={(e) => {
              changeData.setData(e.target.value);
            }}
          />
        </>
      )}
    </div>
  );
}
