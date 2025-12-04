import React, { useState, createContext, useMemo } from "react";
import A_PropsDrilling from "./A_PropsDrilling";
import B_UseContext_Hooks from "./B_UseContext_Hooks";
import B_UseContext_Tag from "./B_UseContext_Tag";
import B_UseObject from "./B_UseObject";
import B_DefaultContext from "./B_DefaultContext";
//context import하기
import { ChangeContext, ContextTest } from "./resources/Context";
import C_ChangeContextValue from "./C_ChangeContextValue";
import C_ChangeContextValue2 from "./C_ChangeContextValue2";

//모듈화 provider가져오기
import {
  MyContextPropsProvider,
  MyContextProvider,
} from "./resources/myprovider";
import D_ModuleContextUse from "./D_ModuleContextUse";
import D_Child from "./child/D_Child";

export default function ContextContainer() {
  const [data, setData] = useState("");
  const [count, setCount] = useState(0);
  // const renderingTest = {data : "랜더링테스트데이터"};
  //useMemo()이용해서 랜더링최적화
  const renderingTest = useMemo(() => "랜더링테스트", []);

  return (
    <div>
      <h2>Context활용하기</h2>
      <p>전역에서 데이터를 공유하는 기능</p>
      <h2>props drilling문제</h2>
      <A_PropsDrilling data={data}>Props데이터 Drilling문제</A_PropsDrilling>
      <input
        type="text"
        onChange={(e) => {
          setData(e.target.value);
        }}
      />
      <h2>Context적용하기</h2>
      <p>
        별도의 js파일을 생성해서 활용. 기본 모듈로 createContext()함수를 이용해서
        관리할 데이터를 생성. 생성한 createContext()의 이름에 Provider를 붙여서
        태그로 선언 예) {"<"}Test.Provider value="사용할 데이터 설정 "{">"};
      </p>

      <h3>기본값 전달하기</h3>
      <p>사용할 컴포넌트에서 createContext()로 생성한 객체를 import해서 사용함.</p>
      <ContextTest.Provider value="basicData">
        <B_UseContext_Tag>태그 방식으로 Context이용하기</B_UseContext_Tag>
        <B_UseContext_Hooks>함수방식으로 Context이용하기</B_UseContext_Hooks>
      </ContextTest.Provider>

      <h3>객체 전달하기</h3>
      <ContextTest.Provider value={{ name: "유병승", age: 19 }}>
        <B_UseObject></B_UseObject>
      </ContextTest.Provider>

      <h3>설정된 기본값 이용하기</h3>
      <p>
        Provider를 선언하지 않고 context값을 가져오면 생성시 인수로 설정한
        기본값을 가져와 사용함
      </p>
      <B_DefaultContext></B_DefaultContext>

      <h3>context데이터 수정하기</h3>
      <p>props의 값이나 리터럴 값을 설정하면 수정이 불가능함</p>
      <h4>리터럴값을 데이터로 설정한 것</h4>
      <ChangeContext.Provider value="123">
        <C_ChangeContextValue />
        <C_ChangeContextValue2 />
      </ChangeContext.Provider>
      <h3>state와 연결한 데이터 수정</h3>
      <p>
        useState와 context를 연결하면 반응성이 생겨 데이터가 변경되면 리랜더링됨
      </p>
      <ChangeContext.Provider value={{ data: data, setData: setData }}>
        <C_ChangeContextValue />
        <C_ChangeContextValue2 />
      </ChangeContext.Provider>

      <h3>context 모듈화 하기</h3>
      <p>
        resouces폴더에 provider를 반환하는 함수와 consumer함수를 선언한 jsx를
        생성하고 provider를 불러와서 사용
      </p>
      <h4>기본값으로 context데이터 공유하기</h4>
      <MyContextProvider>
        <D_ModuleContextUse></D_ModuleContextUse>
      </MyContextProvider>
      <h4>props로 Context데이터 커스터마이징하고 처리하기</h4>
      <MyContextPropsProvider id="bslove" pw="bslove1234">
        <D_ModuleContextUse isUpdate={false}>수정 불가능</D_ModuleContextUse>
        <D_ModuleContextUse isUpdate={true}>수정하기</D_ModuleContextUse>
      </MyContextPropsProvider>

      <h3>랜더링 문제</h3>
      <p>
        context와 상관이 없는 데이터를 변경하더라도 같이 랜더링되는 문제가
        발생함
        <br />
        아래 예제에서 button에서 증가하는 데이터와 context와 상관이없는데
        D_Child도 같이 리랜더링되어 버림
        <br />
        최적화방법 context데이터에 useMemo()를 이용해서 처리
      </p>
      <ContextTest.Provider value={renderingTest}>
        <button
          onClick={() => {
            setCount((prev) => prev + 1);
          }}
        >
          증가 : {count}
        </button>
        <D_Child></D_Child>
      </ContextTest.Provider>
    </div>
  );
}
