import React from "react";
import A_PropsClass from "./props/A_PropsClass";
import A_PropsFunction from "./props/A_PropsFunction";
import B_PropManyData from "./props/B_PropManyData";
import B_ProManyData2 from "./props/B_ProManyData2";
import C_StyleProp from "./props/C_PropsStyle";
import C_PropsClassStyle2 from "./props/C_PropsClassStyle2";
import D_ChildComponent2 from "./props/D_ChildComponent2";
import D_ChildComponent from "./props/D_ChildComponent";
import CommonContainerComponent from "./props/sample/CommonContainerComponent";
import A_StateBasicClass from "./state/A_StateBasicClass";
import A_StateBasicFunction from "./state/A_StateBasicFunction";
import B_StateObjectUse from "./state/B_StateObjectUse";
import C_StateSendUse from "./state/C_StateSendUse";
import B_StateObjectUse2 from "./state/B_StateObjectUse2";
import D_LoadingTest from "./state/D_LoadingTest";
import D_EffectStateComponent from "./state/E_EffectStateComponent";
import ContextContainer from "./context/ContextContainer";
export default function DataContainer() {
  //데이터 전송객체
  const strData = "문자열데이터";
  const numData = 19;
  const arrData = [1, 2, 3, 4, 5];
  const objData = { name: "유병승", age: 19, address: "경기도 시흥시" };
  const funcData = () => {
    alert("prop함수");
  };
  //스타일 적용 객체
  const style1 = { fontSize: "20px", fontWeight: "bolder" };
  const style2 = {
    fontSize: "10px",
    backgroundColor: "lightblue",
    color: "white",
    width: "40%",
  };

  //클래스를 전달해서 적용하기 -> tailwindcss
  //웹폰트를 이용하려면 theme로 등록하고 사용해야함.
  const classStyle = ["font-black", "text-3xl", "text-red-200"];
  const classStyle2 = [
    "font-black",
    "font-center",
    "bg-gradient-to-r from-lime-500 to-white-200",
    "w-fit",
  ];
  return (
    <div>
      <h2>컴포넌트에서 사용하는 데이터</h2>
      <h3>Props데이터활용하기</h3>
      <A_PropsClass title="class props데이터 이용"></A_PropsClass>
      <A_PropsFunction title="function props데이터 이용"></A_PropsFunction>
      <B_PropManyData
        strData={strData}
        numData={numData}
        arrData={arrData}
        objData={objData}
        funcData={funcData}
        isShow={true}
        isHidden={false}
      ></B_PropManyData>
      <B_ProManyData2
        strData={strData}
        numData={numData}
        arrData={arrData}
        objData={objData}
        funcData={funcData}
        isShow={true}
        isHidden={false}
      ></B_ProManyData2>
      <C_StyleProp style={style1}></C_StyleProp>
      <C_StyleProp style={style2}></C_StyleProp>
      <C_PropsClassStyle2 classStyle={classStyle}></C_PropsClassStyle2>
      <C_PropsClassStyle2 classStyle={classStyle2}></C_PropsClassStyle2>
      <h3>컴포넌트의 children props가져오기</h3>
      <p>
        컴포넌트의 시작태그와 끝태그사이에 작성하는 값을 children props이라고
        하고 자식컴포넌트에서 props.children속성으로 데이터를 가져올 수 있음.
      </p>
      <h3>props객체에서 가져오기</h3>
      <D_ChildComponent>기본 Children값 가져오기</D_ChildComponent>
      <D_ChildComponent>{10}</D_ChildComponent>
      <D_ChildComponent>{[1, 2, 3, 4, 5]}</D_ChildComponent>
      <D_ChildComponent>{{ name: "유병승", age: 19 }}</D_ChildComponent>
      <h3>구조분해할당으로 가져오기</h3>
      <D_ChildComponent2>children값</D_ChildComponent2>
      <p>배열, jsx 등이 가능</p>
      <D_ChildComponent2>{[1, 2, 3, 4, 5]}</D_ChildComponent2>
      <D_ChildComponent2>
        <span style={{ color: "blue" }}>나는 어떻게해?</span>
      </D_ChildComponent2>
      <h4>다수 jsx전달하기</h4>
      <p>다수의 jsx를 전달하면 배열방식으로 전달됨</p>
      <D_ChildComponent2>
        <span style={{ color: "blue" }}>나는 어떻게해?</span>
        <p className={classStyle.join(" ")}>두번째는 p태그</p>
        <p className={classStyle2.join(" ")}>세번째는 p태그</p>
      </D_ChildComponent2>

      <h3>공통컴포넌트 만들어 활용하기</h3>
      <p>props데이터를 전달받아 컴포넌트를 구성</p>
      <CommonContainerComponent></CommonContainerComponent>

      <h2>반응성있는 데이터 활용하기</h2>
      <A_StateBasicClass></A_StateBasicClass>
      <A_StateBasicFunction></A_StateBasicFunction>
      <B_StateObjectUse></B_StateObjectUse>
      <B_StateObjectUse2>state를 이용한 회원관리</B_StateObjectUse2>
      <C_StateSendUse>컴포넌트를 나눠서 구현하기</C_StateSendUse>
      <h2>생명주기 함수와 state활용 하기 </h2>
      <D_LoadingTest>state를 이용한 loading화면처리</D_LoadingTest>
      <D_EffectStateComponent></D_EffectStateComponent>

      <h2>Context활용하기</h2>
      <ContextContainer />
    </div>
  );
}
