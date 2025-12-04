import React, { useState, useEffect } from "react";
import E_EffectDebouncing from "./components/E_EffectDebouncing";

//state값이 변경할때마다 특정 로직을 실행하기
export default function D_EffectStateComponent() {
  const [data, setData] = useState();
  const [data2, setData2] = useState();
  useEffect(() => {
    console.log("data가 수정되면 실행");
  }, [data]);
  useEffect(() => {
    console.log("state값이 수정되면 모두 실행");
  });
  const changeData = (e) => {
    const { name, value } = e.target;
    switch (name) {
      case "data":
        setData(value);
        break;
      case "data2":
        setData2(value);
        break;
    }
  };
  return (
    <div>
      <h3>data state가 변경시 특정로직을 실행하기</h3>
      <p>data : {data}</p>
      data 수정 : <input type="text" name="data" onChange={changeData} />
      <p>data2 : {data2}</p>
      data2 수정 : <input type="text" name="data2" onChange={changeData} />
      <h3>디바운싱적용하기를 적용하기</h3>
      <p>조회하는 query를 보낼때 이용</p>
      <E_EffectDebouncing>
        검색어 입력시 검색요청을 하는 컴포넌트
      </E_EffectDebouncing>
    </div>
  );
}
