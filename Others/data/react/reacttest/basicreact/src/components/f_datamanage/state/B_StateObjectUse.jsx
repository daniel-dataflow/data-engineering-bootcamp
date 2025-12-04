import React, { useState } from "react";

export default function B_StateObjectUse() {
  const [arrData, setArrayData] = useState([]);
  const [objData, setObjData] = useState({
    name: "",
    age: 0,
    address: "",
    isActive: false,
  });
  const addArray = (e) => {
    // set메소드를 이용해서 수정
    // State에 값을 추가할때 항상 새로운 값을 추가
    // 비동기로 데이터가 업데이트 되어 이전 값을 가지고 수정할때 유의!
    setArrayData([...arrData, e.target.value]);
  };
  const nameData = (e) => {
    // setObjData({ ...objData, name: e.target.value });
    objDataManage({ name: e.target.value });
  };
  const ageData = (e) => {
    // setObjData({ ...objData, age: e.target.value });
    objDataManage({ age: e.target.value });
  };
  const addressData = (e) => {
    // setObjData({ ...objData, address: e.target.value });
    objDataManage({ address: e.target.value });
  };
  const isActive = (e) => {
    // setObjData({ ...objData, isActive: e.target.value });
    objDataManage({ isActive: e.target.value });
  };
  const objDataManage = (param) => {
    setObjData({ ...objData, ...param });
  };

  //한개로 분할하기
  const objDataHandler = (e) => {
    const { name, value } = e.target;
    setObjData({ ...objData, [name]: value });
  };
  const [count, setCount] = useState(0);
  const incrementNum = () => {
    //2씩 증가해야하는데 1씩증가하고 한개는 생략됨.
    setCount(count + 1);
    setCount(count + 1);
  };
  const incrementNum2 = () => {
    //데이터의 일관성을 유지하면서 변경이 가능해짐.
    //이전데이터를 활용할때 반드시 이 방법을 이용해야 안전함.
    setCount((prev) => prev + 1);
    setCount((prev) => prev + 1);
  };
  return (
    <div>
      <h3>객체, 배열 state활용하기</h3>
      <p>state에 있는 객체를 출력할때는 함수를 이용해서 JSX를 만들어서 출력</p>
      <h4>배열출력</h4>
      <p>{arrData.length == 0 ? "빈배열" : arrData}</p>
      <h4>객체출력</h4>
      <p>
        {Object.keys(objData).length == 0 ? "빈객체" : Object.keys(objData)}
      </p>
      <h4>배열에 값을 추가하기</h4>
      <input type="text" onChange={addArray} />
      <h5>배열 리스트로 출력</h5>
      <ul>
        {arrData.map((d) => (
          <li>{d}</li>
        ))}
      </ul>
      <h5>객체의 key:value를 리스트로 출력</h5>
      <ul>
        {Object.entries(objData).map((e, i) => (
          <li key={i}>
            {e[0]} : {e[1]}
          </li>
        ))}
      </ul>
      <h4>Object데이터 수정하기</h4>
      <p>
        새로운 객체를 생성해서 set메소드를 이용해야함. 원본을 복제한 후 새로운
        객체를 생성해서 활용함
      </p>
      <h4>기본으로 수정하기</h4>
      <input type="text" onChange={nameData} placeholder="이름입력" />
      <br />
      <input type="number" onChange={ageData} placeholder="나이입력" />
      <br />
      <input type="text" onChange={addressData} placeholder="주소입력" />
      <br />
      <label>
        <input type="radio" name="isActive" onChange={isActive} value="true" />
        Yes
      </label>
      <label>
        <input type="radio" name="isActive" onChange={isActive} value="false" />
        No
      </label>
      <h4>한개로 통합하기</h4>
      <input
        type="text"
        name="name"
        onChange={objDataHandler}
        placeholder="이름입력"
      />
      <br />
      <input
        type="number"
        name="age"
        onChange={objDataHandler}
        placeholder="나이입력"
      />
      <br />
      <input
        type="text"
        name="address"
        onChange={objDataHandler}
        placeholder="주소입력"
      />
      <br />
      <label>
        <input
          type="radio"
          name="isActive"
          onChange={objDataHandler}
          value="true"
        />
        Yes
      </label>
      <label>
        <input
          type="radio"
          name="isActive"
          onChange={objDataHandler}
          value="false"
        />
        No
      </label>
      <h3>이전값을 수정하기</h3>
      <p>
        state의 set메소드는 비동기로 데이터를 처리하기 때문에 이전데이터를
        수정할때는 유의해야함.
      </p>
      <h4>이전 값 수정의 문제점</h4>
      <h3>{count}</h3>
      <button onClick={incrementNum}>데이터일관성 깨짐</button>&nbsp;
      <button onClick={incrementNum2}>데이터일관성 유지</button>
    </div>
  );
}
