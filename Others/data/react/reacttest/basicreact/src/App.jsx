import A_ClassComponent from "./components/a_component/A_ClassComponent";
import A_FunctionComponent from "./components/a_component/B_FunctionComponent.js";
import A_LifeCylce_Class from "./components/b_lifecycle/A_LifeCylce_Class.jsx";
import B_LifeCycle_Func from "./components/b_lifecycle/B_LifeCycle_Func.jsx";
import B_JSXBasic from "./components/c_jsx/A_JSXBasic.jsx";
import B_UseJavascript from "./components/c_jsx/B_UseJavascript_Func.jsx";
import C_UseJavascriptFunc from "./components/c_jsx/C_UseJavascriptFunc.jsx";
import D_ConditionTest from "./components/c_jsx/D_ControllerTest.jsx";
import E_ExportData from "./components/c_jsx/E_ExportData.jsx";
import A_BasicStyle from "./components/d_style/A_BasicStyle.jsx";
import B_Header from "./components/d_style/B_Header_Module.jsx";
import B_Footer from "./components/d_style/B_Footer_Module.jsx";
import C_StyledComponent from "./components/d_style/C_StyledComponent.jsx";
import D_TailwindContainer from "./components/d_style/tailwindtest/D_TailwindContainer.jsx";
import EventContainer from "./components/e_event/EventContainer.jsx";
import DataContainer from "./components/f_datamanage/DataContainer.jsx";
import "./components/d_style/tailwindtest/tailwindtest.css";
import HooksContainer from "./components/g_usehooks/HooksContainer.jsx";

function App() {
  const style = {
    padding: "3%",
  };
  return (
    <div style={style}>
      <h1>기본 컴포넌트 선언하기</h1>
      <h1>클래스형 컴포넌트</h1>
      <A_ClassComponent></A_ClassComponent>
      <h1>함수형 컴포넌트</h1>
      <A_FunctionComponent></A_FunctionComponent>
      <h1>생명주기함수</h1>
      {/* <A_LifeCylce_Class></A_LifeCylce_Class> */}
      <B_LifeCycle_Func></B_LifeCycle_Func>
      <h1>JSX사용하기</h1>
      <B_JSXBasic></B_JSXBasic>
      <h1>js구문이용하기</h1>
      <B_UseJavascript></B_UseJavascript>
      <h1>함수활용하기</h1>
      <C_UseJavascriptFunc></C_UseJavascriptFunc>
      <h1>조건문 활용하기</h1>
      <D_ConditionTest></D_ConditionTest>

      <h1>외부js파일이용하기</h1>
      <E_ExportData></E_ExportData>
      <h1>스타일활용하기</h1>
      <A_BasicStyle></A_BasicStyle>

      <h2>전역으로 설정했을때 문제점!</h2>
      <p>컴포넌트를 import한 순서에 따라 css가 적용됨.</p>
      <p>
        모듈로 적용하면 동일한 명칭을 사용해도 react가 컴포넌트별로 class명을
        중복되지 않게 부여해서 처리함
      </p>
      <B_Header></B_Header>
      <B_Footer></B_Footer>

      <h1>스타일 라이브러리 활용하기</h1>
      <C_StyledComponent></C_StyledComponent>
      <h1>tailwind css프레임워크 이용하기</h1>
      <D_TailwindContainer></D_TailwindContainer>
      <h1>이벤트설정하기</h1>
      <EventContainer></EventContainer>
      <h1>데이터 관리하기</h1>
      <DataContainer></DataContainer>

      <h1>Hooks이용하기</h1>
      <HooksContainer></HooksContainer>
    </div>
  );
}

export default App;
