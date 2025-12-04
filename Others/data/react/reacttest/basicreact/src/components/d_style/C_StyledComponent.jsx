import React from "react";
import styled from "styled-components";
import C_StyledOtherComponent from "./C_StyledOtherComponent";

//스타일이 적용된 컴포넌트를 생성 -> 함수형컴포넌트에서 함수외부에 선언해서 활용
const Container = styled.div`
  display: flex;
  justify-content: center;
  width: 70%;
  border: 1px solid red;
`;
//h1태그가 있는 컴포넌트를 생성한것과 동일
const Title = styled.h1`
  font-size: 1.5em;
  text-align: center;
  color: lightgray;
`;
//스타일 재사용하기
//styled()함수방식으로 이용 -> 커링함수로 구현되어 있음
//추가로 설정한 스타일을 적용할때 사용
const ContainerRightBottom = styled(Container)`
  justify-content: right;
  align-items: end;
  height: 80px;
`;
const TitleBlack = styled(Title)`
  background-color: black;
  margin: 0;
`;

const OtherComponentStyle = styled(C_StyledOtherComponent)`
  font-size: 20px;
  background-color: magenta;
  color: lime;
`;
//불가능 styled(Container)는 적용되지 않음
// const OtherComponentStyle2 = styled(Container)(C_StyledComponent)`
//   font-size: 20px;
//   background-color: magenta;
//   color: lime;
// `;
export default function C_StyledComponent() {
  return (
    <div>
      {/* styled로 선언한 변수는 아래 태그를 선언한 컴포넌트와 동일
      <h1
       className="test" .test{}에는 styled.로 설정한 내용이 대입
      >
        이것과 동일하게 생성
      </h1> */}
      <h2>기본 styled적용하기</h2>
      <p>
        styled.태그명`css스타일작성`형식으로 style을 설정한 컴포넌트를 변수로
        저장해서 사용
      </p>
      <Container>
        <Title>나의 타이틀</Title>
      </Container>
      <h2>스타일 재사용하기</h2>
      <p>
        styled(style생성된 컴포넌트)`추사 설정할 style내용` 으로 기존 스타일에
        추가 스타일을 적용할 수 있음
      </p>
      <ContainerRightBottom>
        <TitleBlack>재사용해서 설정한 타이틀</TitleBlack>
      </ContainerRightBottom>

      <h2>생성한 컴포넌트에 적용하기</h2>
      <p>
        생성한 컴포넌트에 적용하려면 props와 같이 사용 styled로 style을 적용하면
        coponent에 className속성으로 데이터가 전달되고 그 데이터를 받아서 처리함
        *props에 대해 배우고 스타일적용
      </p>
      <C_StyledOtherComponent></C_StyledOtherComponent>
      <OtherComponentStyle></OtherComponentStyle>
    </div>
  );
}
