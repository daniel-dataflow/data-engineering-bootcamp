import React from "react";
import D_LayoutComponent from "./D_LayoutComponent";
// import "./tailwindtest.css";
import D_ButtonComponent from "./sample/D_ButtonComponent";
import D_CardComponent from "./sample/D_CardComponent";
import D_InputFormComponent from "./sample/D_InputFormComponent";
import D_NavBarComponent from "./sample/D_NavBarComponent";
import D_SectionContainer from "./sample/D_SectionContainer";
import D_AlertComponent from "./sample/D_AlertComponent";
const basicContainer = "flex flex-col space-y-10";
export default function D_TailwindContainer() {
  return (
    <div>
      <h2 className="text-2xl font-bold p-3 font-goong bg-bs">
        tailwindcss 클래스 적용하기
      </h2>
      <h2>tailwindcss로 만든 샘플 컴포넌트</h2>
      <div className={`${basicContainer}`}>
        <h3>버튼</h3>
        <D_ButtonComponent></D_ButtonComponent>
        <h3>card</h3>
        <D_CardComponent></D_CardComponent>
        <h3>입력창</h3>
        <D_InputFormComponent></D_InputFormComponent>
        <h3>navbar</h3>
        <D_NavBarComponent></D_NavBarComponent>
        <h3>sectioncontainer</h3>
        <D_SectionContainer></D_SectionContainer>
        <h3>alert컴포넌트</h3>
        <D_AlertComponent></D_AlertComponent>
      </div>
      <h2>커스텀 theme작성하기</h2>
      <p>css속성에 @Theme{} 설정을 이용해서 원하는 내용을 커스텀할 수 있음</p>
      <h3 className="bg-bs text-bs2 m-bssize p-bssize rounded-3xl">
        나의 색상 마젠타 라임
      </h3>
    </div>
  );
}
