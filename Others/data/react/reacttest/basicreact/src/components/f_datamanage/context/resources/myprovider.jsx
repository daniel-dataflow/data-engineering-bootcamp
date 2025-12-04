import React, { createContext, useState, useContext } from "react";

// 모듈화 처리하기

// context객체를 생성하기
const MyContext = createContext();

// provider함수 만들기 -> 필료한 jsx파일에서 import해서 태그 방식으로 사용할 수 있게 됨.
// 다른 곳에서 사용할때 MyContextProvider태그의 텍스트 노드부분을 가져와 처리해야하기때문에 children을 전달받음
export const MyContextProvider = ({ children }) => {
  //컨텍스트에서 관리할 데이터 state로 생성
  const [data, setData] = useState({ id: "admin", pw: "1234" });
  return (
    <MyContext.Provider value={{ data, setData }}>
      {children}
    </MyContext.Provider>
  );
};

// consumer함수 함들기
// 생성된 Context를 useContext()를 호출해서 지정된 데이터를 이용할 수 있게 해주는 함수
export const useMyContext = () => {
  const data = useContext(MyContext);
  if (!data) throw new Error("MyContextProvider태그의 자식태그가 아닙니다.");
  return data; //context데이터를 반환
};

//props값을 받아서 context데이터를 설정하기
//props로 설정한 key:value를 useState 기본값으로 생성해서 value로 공유하는 로직
//context가 props로 전달한 데이터를 관리하는 구조
export const MyContextPropsProvider = ({ children, ...providerData }) => {
  const [data, setData] = useState({ ...providerData });
  return (
    <MyContext.Provider value={{ data, setData }}>
      {children}
    </MyContext.Provider>
  );
};
