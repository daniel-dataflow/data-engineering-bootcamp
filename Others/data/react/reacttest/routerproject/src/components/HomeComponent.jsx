import React, { useState } from "react";
import {
  createSearchParams,
  Link,
  useLocation,
  useNavigate,
} from "react-router-dom";
import HeaderComponent from "./common/HeaderComponent";

export default function HomeComponent() {
  const infoUrl = useLocation();
  const infoHandle = () => {
    console.log(infoUrl);
  };

  //querystring이용하기
  const [filterData, setFilerData] = useState({});
  const navigate = useNavigate();
  const useQuery = (e) => {
    const { name, value } = e.target;
    setFilerData((prev) => ({ ...prev, [name]: value }));
  };
  const searchUser = (e) => {
    const query = createSearchParams(filterData);
    navigate({ pathname: "/userquery", search: `?${query}` });
  };
  return (
    <div>
      <HeaderComponent />
      <h2>메인화면</h2>
      <p>hooks가 주는 정보 확인하기</p>
      <h3>useLocation() : url주소에 대한 정보를 가져올 수 있음</h3>
      <ul>
        <li>pathname : URL경로</li>
        <li>search : ?뒤 문자열 -> querystringdata</li>
        <li>
          hash : url주소의 #뒤에 문자열 해쉬문자 -> 서버에 전송되지않고
          프론트엔드에서 활용하는 정보를 주는 값(위경도표시, 사용자스타일설정,
          검색필터 등)
        </li>
        <li>
          state : navigate("url",{`{state객체}`})함수나 Link컴포넌트에서
          state속성으로 전달한 상태객체
        </li>
        <li>key : location객체의 고유값 -> </li>
      </ul>
      <ul>
        {Object.entries(infoUrl).map((url) => (
          <li key={url[0]}>{`${url[0]} : ${url[1]}`}</li>
        ))}
      </ul>
      <Link to="/?name=유병승#12345" state={{ apikey: "#123411" }}>
        querystring,hash,sate값 전송하고 데이터 확인하기
      </Link>
      <br />
      <button onClick={infoHandle}>userLocation() 정보확인</button>
      <h3>queryString이용하기</h3>
      나이 : <input type="text" name="age" onChange={useQuery} />
      이름 : <input type="text" name="name" onChange={useQuery} />
      <button onClick={searchUser}>검색</button>
    </div>
  );
}
