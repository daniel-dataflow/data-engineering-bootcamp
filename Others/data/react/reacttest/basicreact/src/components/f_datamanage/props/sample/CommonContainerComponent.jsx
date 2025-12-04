import React from "react";
import CardComponent from "./CardComponent";
import logo from "@/assets/react.svg";
import baiscAvatar from "@/assets/basic-avatar.png";
import profile from "@/assets/profile.png";
import AvatarComponent from "./AvatarComponent";
import ModalComponent from "./ModalComponent";
import ButtonComponent from "./ButtonComponent";
import TableComponent from "./TableComponent";

export default function CommonContainerComponent() {
  const header = ["번호", "제목", "작성자", "작성일"];
  const body = [
    { no: 1, title: "첫번째 제목", writer: "user1", writeDate: new Date() },
    { no: 2, title: "두번째 제목", writer: "user2", writeDate: new Date() },
    { no: 3, title: "세번째 제목", writer: "user3", writeDate: new Date() },
    { no: 4, title: "네번째 제목", writer: "user4", writeDate: new Date() },
    { no: 5, title: "다섯번째 제목", writer: "user5", writeDate: new Date() },
  ];
  return (
    <div>
      <h3>CardComponent이용하기</h3>
      <div className="grid lg:grid-cols-5 sm:grid-cols-3 gap-10">
        <CardComponent title="첫번째 카드">첫번째 카드컴포넌트</CardComponent>
        <CardComponent title="두번째 카드" imgSrc={logo} hightlight={true}>
          두번째 카드컴포넌트
        </CardComponent>
        <CardComponent title="세번째 카드" imgSrc={logo} hightlight={false}>
          세번째 카드컴포넌트
        </CardComponent>
        <CardComponent title="네번째 카드" imgSrc={logo} hightlight={false}>
          네번째 카드컴포넌트
        </CardComponent>
        <CardComponent title="다섯번째 카드" imgSrc={logo} hightlight={false}>
          다섯번째 카드컴포넌트
        </CardComponent>
      </div>
      <h3>아바타 컴포넌트</h3>
      <div>
        <AvatarComponent
          size="100"
          src={baiscAvatar}
          alt="기본프로필"
        ></AvatarComponent>
        <AvatarComponent
          size="200"
          src={profile}
          alt="나의 프로필"
        ></AvatarComponent>
      </div>
      <h3>버튼 적용하기</h3>
      <div className="grid grid-cols-5 gap-3">
        <ButtonComponent
          onClick={() => {
            alert("기본버튼클릭함");
          }}
        >
          기본버튼
        </ButtonComponent>
        <ButtonComponent
          variant="secondary"
          onClick={() => {
            alert("커스텀버튼 클릭");
          }}
        >
          secondary
        </ButtonComponent>
        <ButtonComponent
          variant="warn"
          onClick={() => {
            alert("커스텀버튼 클릭");
          }}
        >
          warn
        </ButtonComponent>
        <ButtonComponent
          variant="danger"
          onClick={() => {
            alert("커스텀버튼 클릭");
          }}
        >
          error
        </ButtonComponent>
      </div>
      <h3>데이터를 처리하는 table만들기</h3>
      <TableComponent header={header} body={body}></TableComponent>
    </div>
  );
}
