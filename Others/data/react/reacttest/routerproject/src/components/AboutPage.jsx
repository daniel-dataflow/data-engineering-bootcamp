import React from "react";
import HeaderComponent from "./common/HeaderComponent";
import { useLocation, useSearchParams, useNavigate } from "react-router-dom";
export default function AboutPage() {
  const [searchParams, setSearParam] = useSearchParams();
  //코드로 페이지 전환요청을 하는 hook
  const navigate = useNavigate();
  return (
    <div>
      <HeaderComponent />
      <h2>Link로 전환된 내용 확인하기</h2>
      <div>
        <h3>전송한 데이터 각각 확인하기</h3>
        <p>pathname : {useLocation().pathname}</p>
        <p>
          querystring : {JSON.stringify(Object.fromEntries([...searchParams]))}
        </p>
        <p>hash : {useLocation().hash}</p>
        <p>state : {JSON.stringify(useLocation().state)}</p>
      </div>
      <button
        onClick={() => {
          // 매개변수에 특정 주소나 인덱스번호를 설정(-1은 뒤로,1은 앞으로)
          navigate(-1); //replace=true로 설정해서 옮긴 것은 페이지 뒤로가기가 되지 않음
        }}
      >
        뒤로 이동하기
      </button>
    </div>
  );
}
