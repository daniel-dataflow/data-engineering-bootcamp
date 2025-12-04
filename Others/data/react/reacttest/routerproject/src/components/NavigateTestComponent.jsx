import React from "react";
import { useNavigate } from "react-router-dom";

export default function NavigateTestComponent() {
  //useNavigate()를 이용해서 이벤트와 연결해서 페이지 이동하기
  //먼저 객체를 생성하기
  const navigate = useNavigate();
  const movePath = () => {
    navigate("/about");
  };
  const moveIndex = () => {
    //히스토리에 있는 값을 기준으로 동작 -> 몇칸 이동을 설정
    console.log(window.history.length);
    // navigate(-1);//뒤로가기 뒤로 한칸
    navigate(1); //앞으로 가기 앞으로 한칸
  };
  const moveQueryStringHash = () => {
    // navigate("/about?name=유병승");
    navigate({ pathname: "/about", search: "?name=유병승", hash: "#111222" });
  };
  const moveState = () => {
    //두번째 매개변수 객체에 state키로 설정한 데이터를 전달
    navigate("/about", { state: { id: "admin" } });
  };
  return (
    <div>
      <h3>태그에 클릭 이벤트에서 페이지전환요청 보내기</h3>
      <div>
        <button onClick={movePath}>경로를 설정해서 이동하기</button>
        <button onClick={moveIndex}>인덱스로 이동하기</button>
        <button onClick={moveQueryStringHash}>쿼리스트링,해쉬 같이 이동</button>
        <button onClick={moveState}>state같이 이동</button>
      </div>
    </div>
  );
}
